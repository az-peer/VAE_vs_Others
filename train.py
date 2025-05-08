import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from model import VariationalAutoEncoder
import numpy as np


device = torch.device("cpu")
input_dim = 784
hidden_dim = 200
z_dim = 20
num_epochs = 60
batch_size = 128
karpathy_constant = 3e-4



dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
random_indices = np.random.choice(len(dataset), size=5000, replace=False)
subset_train_dataset = Subset(dataset, random_indices)

train_loader = DataLoader(
    dataset=subset_train_dataset,batch_size=batch_size,shuffle = True
    )

model = VariationalAutoEncoder(input_dim=input_dim, 
                               hidden_dim=hidden_dim,
                               z_dim=z_dim)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = karpathy_constant)
loss_fn = nn.BCELoss(reduction='sum')


for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))

    epoch_loss = 0

    # not using labels for now 
    for i, (x, _) in loop:
        x = x.to(device).view(x.shape[0], input_dim)

        x_reconstructed, mu, logvar = model(x)

        recontruction_loss = loss_fn(x_reconstructed, x)

        # kl divergence 
        # minimizing the same as negative
        # pushes towards gaussian
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recontruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.2f}")


# save the model
torch.save(model.state_dict(), "vae_model.pth")

