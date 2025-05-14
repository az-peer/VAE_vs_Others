import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
import clip
import os

# ==========================
# Parameters (Edit as needed)
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
num_epochs = 50
image_size = 28
learning_rate = 1e-4
guidance_scale = 2.0
output_dir = "/content/drive/My Drive/VAE_vs_Others/azfals_shit"
fashion_labels = [
    "A short-sleeved cotton crewneck shirt", "Trouser", "A thick wool sweater", "Dress", "A long overcoat with buttons",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ==========================
# Load CLIP model
# ==========================
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def get_text_embedding(prompt):
    with torch.no_grad():
        text = clip.tokenize([prompt]).to(device)
        text_features = clip_model.encode_text(text).float()
    return text_features / text_features.norm(dim=-1, keepdim=True)

# ==========================
# Load Dataset
# ==========================
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
import torch.serialization
torch.serialization.add_safe_globals(["Subset"])
data_path = "/content/drive/My Drive/VAE_vs_Others/semi_imbalanced_5000.pt"
subset = torch.load(data_path, weights_only=False)
#images, labels = torch.load(data_path) 
#dataset = torch.utils.data.TensorDataset(images, labels)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

images = torch.stack([subset[i][0] for i in range(len(subset))])
images = (images - 0.5) / 0.5

labels = torch.tensor([subset[i][1] for i in range(len(subset))])
dataset = torch.utils.data.TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================
# Define DDPM Components
# ==========================
model = UNet2DModel(
    sample_size=image_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
   # class_embed_type="projection",
    num_class_embeds=10
).to(device)
print(model.parameters())
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# ==========================
# Training Loop
# ==========================
model.train()
for epoch in range(num_epochs):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for batch in pbar:
        x = batch[0].to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x.shape[0],), device=device).long()
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        class_labels = batch[1].to(device)
        noise_pred = model(noisy_x, timesteps, class_labels=class_labels).sample

        loss = loss_fn(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
ckpt_path = "./model.pth"          # pick any path you like
#torch.save(model.state_dict(), ckpt_path)
#print(f"Model saved to {ckpt_path}")
torch.save(model.state_dict(), "/content/drive/My Drive/VAE_vs_Others/diffusion_semi_5000.pth")
print("Model saved to Google Drive at VAE_vs_Others/diffusion_semi_5000.pth")

# ==========================
# Guided Generation: One Per Class
# ==========================
model.eval()
os.makedirs(output_dir, exist_ok=True)

@torch.no_grad()
def generate_per_class():
    for i in range(100):
        for class_idx, class_name in enumerate(fashion_labels):
            latents = torch.randn((1, 1, image_size, image_size), device=device)
            for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                class_tensor = torch.tensor([class_idx], device=device)
                noise_pred = model(latents, t_batch, class_labels=class_tensor).sample
                latents = noise_scheduler.step(noise_pred, t_batch, latents).prev_sample

            safe_class_name = class_name.replace("/", "_")
            filename = f"{i}_{class_idx}_{safe_class_name}.png"
            print(os.path.join(output_dir, filename))
            save_image((latents + 1) / 2.0, os.path.join(output_dir, filename))
            print(f"Generated ({i}): {class_name}")

generate_per_class()
print(f"Images saved to {output_dir}")
