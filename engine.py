import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from model import VariationalAutoEncoder
import numpy as np

###################################### ADDING NOISE ######################################
def add_noise(x, noise_factor=0.3):
    noisy = x + noise_factor * torch.randn_like(x)
    noisy = torch.clamp(noisy, 0., 1.)
    return noisy

###################################### TRAINING ######################################
def train_model(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs,
    input_dim,
    add_noise_flag=False,
    noise_factor=0.3,
    save_path="vae_model.pth",
    cnn_flag = False
):
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        epoch_loss = 0
        if not cnn_flag:
            for i, (x, _) in loop:
                x = x.to(device).view(x.shape[0], input_dim)
                if not add_noise_flag:
                    print("Running Normal VAE")
                    x_reconstructed, mu, logvar = model(x)
                else:
                    print("Running DVAE")
                    x_noisy = add_noise(x, noise_factor)
                    x_reconstructed, mu, logvar = model(x_noisy)
                recontruction_loss = loss_fn(x_reconstructed, x)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recontruction_loss + kl_div
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())
        else:
            for i, (x, y) in loop:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())


        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.2f}")
    torch.save(model.state_dict(), save_path)

###################################### INFERENCE ######################################
# Define inference function
def inference(model, test_loader, device, num_samples=8, 
              save_path="inference_reconstruction.png", dataset_save_path="generated_dataset.pt",
              cnn_flag=False):
    if cnn_flag:
        # For CNN, we don't need to save reconstructions
        print("CNN model - skipping reconstruction visualization")
        return
    
    # Get one batch
    test_batch = next(iter(test_loader))[0].to(device)
    test_batch_flat = test_batch.view(test_batch.size(0), -1)
    
    # Inference
    with torch.no_grad():
        recon_batch, _, _ = model(test_batch_flat)
    
    # Reshape for visualization
    recon_batch = recon_batch.view(-1, 1, 28, 28)
    original = test_batch.view(-1, 1, 28, 28)
    
    # Concatenate and save
    comparison = torch.cat([original, recon_batch])
    save_image(comparison, save_path, nrow=num_samples)
    
    # Store generated images in a new dataset
    generated_dataset = TensorDataset(recon_batch)
    torch.save(generated_dataset, dataset_save_path)
    print(f"Generated dataset saved to {dataset_save_path}")

def evaluate_cnn(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    return accuracy