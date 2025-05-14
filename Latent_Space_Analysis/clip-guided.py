# ─────────────────────────────── imports ────────────────────────────────────
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler  # Changed import
import clip
from diffusers import UNet2DModel

# ──────────────────────── hyper-parameters / paths ──────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
num_epochs = 21
image_size = 28
lr = 1e-4
guidance_scale = 2.0
cf_dropout = 0.10
output_dir = "./clip_guided_ddpm_output"
data_path = "/content/drive/My Drive/VAE_vs_Others/train_dataset.pt"

fashion_captions = [
    "A short-sleeved cotton crewneck shirt", "Trouser",
    "A thick wool sweater", "Dress", "A long overcoat with buttons",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ─────────────────────────────── CLIP setup ────────────────────────────────
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_dim = 512

@torch.no_grad()
def txt_emb(sentences):
    tok = clip.tokenize(sentences).to(device)
    feats = clip_model.encode_text(tok).float()
    return feats / feats.norm(dim=-1, keepdim=True)

label2clip = torch.vstack([txt_emb([cap]) for cap in fashion_captions])  # [10,512]
null_emb = torch.zeros(clip_dim, device=device)

# ───────────────────────────── dataset load ────────────────────────────────
# ... (unchanged dataset loading code) ...
import torch.serialization
torch.serialization.add_safe_globals(["Subset"])
data_path = "/content/drive/My Drive/VAE_vs_Others/high_imbalance_train.pt"
subset = torch.load(data_path, weights_only=False)
#images, labels = torch.load(data_path) 
#dataset = torch.utils.data.TensorDataset(images, labels)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

images = torch.stack([subset[i][0] for i in range(len(subset))])
images = (images - 0.5) / 0.5

labels = torch.tensor([subset[i][1] for i in range(len(subset))])
dataset = torch.utils.data.TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ───────────────────────────── model + noise ───────────────────────────────
unet = UNet2DConditionModel(
    sample_size=image_size,
    in_channels=1,               # grayscale
    out_channels=1,              # output is also grayscale
    layers_per_block=2,
    block_out_channels=(64, 128, 256),
    down_block_types=(
      "DownBlock2D",
      "DownBlock2D",
      "AttnDownBlock2D",
    ),
    up_block_types=(
      "AttnUpBlock2D",
      "UpBlock2D",
      "UpBlock2D",
    ),
    cross_attention_dim=clip_dim  # length of your CLIP embeddings
).to(device)

sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
opt = torch.optim.Adam(unet.parameters(), lr=lr)
mse = nn.MSELoss()

# ─────────────────────────────── training ──────────────────────────────────
unet.train()
for epoch in range(1, num_epochs+1):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
    for x, y in pbar:
        x = x.to(device)
        n = torch.randn_like(x)
        t = torch.randint(0, sched.config.num_train_timesteps,
                         (x.size(0),), device=device).long()
        x_noisy = sched.add_noise(x, n, t)

        # Classifier-free dropout with CLIP embeddings
        cond_emb = label2clip[y].to(device)
        mask = (torch.rand(cond_emb.size(0), device=device) < cf_dropout).unsqueeze(1)
        class_emb = torch.where(mask, null_emb, cond_emb)

        # Modified forward pass with dual conditioning
        eps_pred = unet(
  x_noisy,
  t,
  encoder_hidden_states=cond_emb.unsqueeze(1)  # [B, 1, 512]
).sample

        loss = mse(eps_pred, n)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

# ─────────────────────────────── sampling ──────────────────────────────────
try:
    from google.colab import drive
    drive.mount('/content/drive')
    output_dir = '/content/drive/My Drive/VAE_vs_Others/guided_ddpm_per_class_output'
except ImportError:
    # Not in Colab—save locally
    output_dir = './guided_ddpm_per_class_output'

import os
os.makedirs(output_dir, exist_ok=True)

@torch.no_grad()
def generate(prompt: str, w: float = guidance_scale,
             steps: int = sched.config.num_train_timesteps):
    e_cond = txt_emb([prompt]).to(device)
    e_un   = null_emb.unsqueeze(0)
    latents = torch.randn((1,1,image_size,image_size), device=device)
    for t in reversed(range(steps)):
        tt = torch.full((1,), t, device=device, dtype=torch.long)
        eps_c = unet(latents, tt, encoder_hidden_states=e_cond.unsqueeze(1)).sample
        eps_u = unet(latents, tt, encoder_hidden_states=e_un.unsqueeze(1)).sample
        eps_h = eps_u + w * (eps_c - eps_u)
        latents = sched.step(eps_h, tt, latents).prev_sample
    return (latents + 1) / 2.0

fashion_labels = [
    "cotton crewneck T‑shirt",
    "tailored trousers",
    "thick wool pullover sweater",
    "light summer dress",
    "button‑front long overcoat",
    "leather sandal",
    "classic dress shirt",
    "athletic sneaker",
    "everyday shoulder bag",
    "leather ankle boot"
]

for caption in fashion_labels:
    img = generate(caption)
    safe_name = caption.replace("/", "_").replace(" ", "_")
    path = os.path.join(output_dir, f"{safe_name}.png")
    save_image(img, path)
    print(f"Generated: {caption} → {path}")
