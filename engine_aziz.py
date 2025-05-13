import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from model import VariationalAutoEncoder
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

###################################### ADDING NOISE ######################################
def add_noise_or_augmentation(x, noise_types=['gaussian'], noise_factor=0.3):
    """Apply multiple types of noise/augmentation to input"""
    x_noisy = x
    
    for noise_type in noise_types:
        if noise_type == 'gaussian':
            noise = torch.randn_like(x_noisy) * noise_factor
            x_noisy = torch.clamp(x_noisy + noise, 0., 1.)
        
        elif noise_type == 'salt_pepper':
            noise = torch.rand_like(x_noisy)
            salt = (noise > 1 - noise_factor/2).float()
            pepper = (noise < noise_factor/2).float()
            x_noisy = torch.clamp(x_noisy * (1 - salt - pepper) + salt, 0., 1.)
        
        elif noise_type == 'speckle':
            noise = torch.randn_like(x_noisy) * noise_factor
            x_noisy = torch.clamp(x_noisy * (1 + noise), 0., 1.)
        
        elif noise_type in ['rotation', 'brightness', 'contrast', 'blur']:
            # Reshape to image format for these transformations
            batch_size = x_noisy.shape[0]
            x_temp = x_noisy.view(batch_size, 1, 28, 28)
            
            if noise_type == 'rotation':
                angle = noise_factor * 30
                x_temp = TF.rotate(x_temp, angle)
            
            elif noise_type == 'brightness':
                factor = 1.0 + (noise_factor * random.uniform(-1, 1))
                x_temp = TF.adjust_brightness(x_temp, factor)
            
            elif noise_type == 'contrast':
                factor = 1.0 + (noise_factor * random.uniform(-1, 1))
                x_temp = TF.adjust_contrast(x_temp, factor)
            
            elif noise_type == 'blur':
                kernel_size = int(3 + noise_factor * 4)
                kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                x_temp = TF.gaussian_blur(x_temp, kernel_size, sigma=noise_factor)
            
            # Reshape back to original format
            x_noisy = x_temp.view(batch_size, -1)
    
    return x_noisy

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
    noise_types=['gaussian'],
    save_path="vae_model.pth",
    cnn_flag=False,
    noisy_classes=None,
    experiment_name="experiment"
):
    # Initialize loss tracking
    epoch_losses = []
    noise_type_str = '_'.join(noise_types) if noise_types else 'none'
    
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        epoch_loss = 0

        if not cnn_flag:
            # Print model type and classes at start of training
            if epoch == 0:
                if add_noise_flag and noisy_classes is not None:
                    print(f"Running DVAE for classes {noisy_classes}")
                    print(f"Using noise types: {noise_types}")
                else:
                    print("Running Normal VAE")

            for i, (x, y) in loop:
                x = x.to(device).view(x.shape[0], input_dim)
                
                if add_noise_flag and noisy_classes is not None:
                    x = x.requires_grad_(True)
                    if i == 0:  # Print grad info for first batch
                        print(f"x_noisy requires_grad: {x.requires_grad}")
                    
                    # Create mask for DVAE samples
                    noise_mask = torch.tensor([label in noisy_classes for label in y]).to(device)
                    x_noisy = x.clone()
                    
                    # Apply noise to DVAE samples only
                    dvae_samples = x[noise_mask]
                    if len(dvae_samples) > 0:  # Only if we have DVAE samples in this batch
                        dvae_samples = add_noise_or_augmentation(dvae_samples, noise_types, noise_factor)
                        x_noisy[noise_mask] = dvae_samples
                    
                    x_input = x_noisy
                else:
                    x_input = x

                # Forward pass
                x_reconstructed, mu, logvar = model(x_input)
                
                # Loss calculation
                reconstruction_loss = loss_fn(x_reconstructed, x)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + kl_div

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        # Print epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        epoch_losses.append({
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
            'noise_types': noise_type_str
        })
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    
    # Return losses for plotting
    return pd.DataFrame(epoch_losses)

def plot_noise_comparison(all_losses, save_path='Results/loss_comparison.png'):
    """Plot loss curves for different noise types"""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create plot
    for noise_type in all_losses['noise_types'].unique():
        data = all_losses[all_losses['noise_types'] == noise_type]
        plt.plot(data['epoch'], data['loss'], label=noise_type, marker='o', markersize=3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Loss Comparison Across Different Noise Types')
    plt.legend(title='Noise Types')
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

###################################### INFERENCE ######################################
# Define inference function
def inference(model, test_loader, device, num_samples=8, 
              save_path="inference_reconstruction.png", dataset_save_path="generated_dataset.pt",
              cnn_flag=False):
    if cnn_flag:
        # For CNN, we don't need to save reconstructions
        print("CNN model - skipping reconstruction visualization")
        return
    
    # Get one batch and limit to num_samples
    test_batch = next(iter(test_loader))[0][:num_samples].to(device)  # Only take num_samples images
    test_batch_flat = test_batch.view(test_batch.size(0), -1)
    
    # Inference
    with torch.no_grad():
        recon_batch, _, _ = model(test_batch_flat)
    
    # Reshape for visualization
    recon_batch = recon_batch.view(-1, 1, 28, 28)
    original = test_batch.view(-1, 1, 28, 28)
    
    # Concatenate and save
    comparison = torch.cat([original, recon_batch])
    save_image(comparison, save_path, nrow=num_samples)  # Changed to num_samples*2 to show pairs in rows
    
    # Store generated images in a new dataset
    generated_dataset = TensorDataset(recon_batch)
    torch.save(generated_dataset, dataset_save_path)
    print(f"Generated dataset saved to {dataset_save_path}")

def evaluate_cnn(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10  # For each class
    class_total = [0] * 10    # For each class
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            
            # Update total and correct counts
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            # Update per-class counts
            for i in range(y.size(0)):
                label = y[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    # Calculate overall accuracy
    accuracy = correct / total
    
    # Print overall accuracy
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Create results dictionary for DataFrame
    results = {
        'Class': list(range(10)),
        'Correct': class_correct,
        'Total': class_total,
        'Accuracy': [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
    }
    
    # Add overall accuracy to results
    results['Overall_Accuracy'] = accuracy
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy = class_correct[i] / class_total[i]
            print(f"Class {i}: {class_accuracy:.2%} ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"Class {i}: No samples")
    
    return results_df

def train_gan(model, train_loader, optimizers, criterion, device, num_epochs):
    """Train Conditional GAN with advanced techniques"""
    optimizer_G, optimizer_D = optimizers
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, num_epochs)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, num_epochs)
    
    for epoch in range(num_epochs):
        for batch_idx, (real_images, labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.view(-1, 784).to(device)
            labels = labels.to(device)
            
            # Dynamic label smoothing
            real_labels = torch.ones(batch_size, 1).to(device) * 0.9
            fake_labels = torch.zeros(batch_size, 1).to(device) * 0.1
            
            # Train Discriminator
            for _ in range(1):  # Optional: multiple D updates
                optimizer_D.zero_grad()
                
                # Real images
                d_real = model.discriminator_forward(real_images, labels)
                d_real_loss = criterion(d_real, real_labels)
                
                # Fake images
                noise = torch.randn(batch_size, model.latent_dim).to(device)
                fake_images = model.generator_forward(noise, labels)
                d_fake = model.discriminator_forward(fake_images.detach(), labels)
                d_fake_loss = criterion(d_fake, fake_labels)
                
                # Gradient penalty (optional)
                alpha = torch.rand(batch_size, 1).to(device)
                interpolated = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)
                d_interpolated = model.discriminator_forward(interpolated, labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                optimizer_D.step()

            # Train Generator
            for _ in range(2):  # Multiple G updates
                optimizer_G.zero_grad()
                noise = torch.randn(batch_size, model.latent_dim).to(device)
                fake_images = model.generator_forward(noise, labels)
                d_fake = model.discriminator_forward(fake_images, labels)
                g_loss = criterion(d_fake, real_labels)
                g_loss.backward()
                optimizer_G.step()

        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

def train_vqvae(model, train_loader, optimizer, device, num_epochs):
    """
    Train VQ-VAE model
    Args:
        model: VQ-VAE model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on
        num_epochs: Number of epochs to train
    Returns:
        lists of losses for plotting
    """
    recon_loss_log = []
    qv_loss_log = []
    train_loss_log = []

    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        
        for i, (x, y) in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}")):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            recon_data, vq_loss, quantized = model(x.view(x.shape[0], -1))
            recon_data = recon_data.view(recon_data.shape[0], 1, 28, 28)

            # Calculate losses
            recon_loss = (recon_data - x).pow(2).mean()
            loss = vq_loss + recon_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log losses
            recon_loss_log.append(recon_loss.item())
            qv_loss_log.append(vq_loss.item())
            train_loss += recon_loss.item()

        # Log average loss for epoch
        avg_loss = train_loss / len(train_loader)
        train_loss_log.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f}, '
                  f'VQ Loss: {vq_loss.item():.4f}, '
                  f'Recon Loss: {recon_loss.item():.4f}')

    return {
        'recon_loss': recon_loss_log,
        'vq_loss': qv_loss_log,
        'train_loss': train_loss_log
    }

def train_biggan(model, train_loader, optimizers, device, num_epochs):
    """
    Train BigGAN model
    Args:
        model: BigGAN model
        train_loader: DataLoader for training data
        optimizers: (optimizer_G, optimizer_D) tuple
        device: Device to train on
        num_epochs: Number of epochs to train
    """
    optimizer_G, optimizer_D = optimizers
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}")):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real images
            d_real = model.discriminator_forward(real_images, labels)
            d_real_loss = criterion(d_real, torch.ones_like(d_real))

            # Fake images
            z = torch.randn(batch_size, model.latent_dim).to(device)
            fake_images = model.generator_forward(z, labels)
            d_fake = model.discriminator_forward(fake_images.detach(), labels)
            d_fake_loss = criterion(d_fake, torch.zeros_like(d_fake))

            # Combined D loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_fake = model.discriminator_forward(fake_images, labels)
            g_loss = criterion(g_fake, torch.ones_like(g_fake))
            g_loss.backward()
            optimizer_G.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
                