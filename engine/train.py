import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from ..models.VAE_CNN import VariationalAutoEncoder
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

###################################### TRAINING VAE and DVAE ######################################
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

############################################### VQ-VAE ##############################################################################
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

######################################################## GAN ################################################################
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
    