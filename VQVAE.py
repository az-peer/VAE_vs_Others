import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
import torch.distributions as Categorical
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from model import VariationalAutoEncoder, CNN
import numpy as np
import argparse
from engine import train_model, inference, evaluate_cnn


class VectorQuantizer(nn.Module):
    # commitment cost is a hyperparameter 
    # part of the loss 
    def __init__(self, code_book_dim, embedding_dim, commitment_cost):
        super().__init__()
        self.code_book_dim = code_book_dim
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(code_book_dim, embedding_dim)
        # randomly initialize codebook
        self.embedding.weight.data.uniform_(-1/code_book_dim, 1/code_book_dim)

    def forward(self, inputs):
        # this flips the image from CxHxW to HxWxC
        # check if this hold for our problem
        inputs = inputs.permute(0, 2, 3, 1).contiguous() # make sure this works for us PRINT THE SIZE WE WANT THIS TO BE BxHxWxC 
        input_shape = inputs.shape

        flat_input = inputs.view(-1, 1, self.embedding_dim)

        # Calculate the distance between each embedding and each codebook vector
        distances = (flat_input - self.embedding.weight.unsqueeze(0)).pow(2).mean(2)

        # Find the closest codebook vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) 

        # get the quantized vector 
        quantized = self.embedding(encoding_indices).view(input_shape)

        # Create loss that pulls encoder embeddings and codebook vector selected
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Reconstruct quantized representation using the encoder embeddings to allow for 
        # backpropagation of gradients into encoder
        if self.training:
            quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices.reshape(input_shape[0], -1)
    
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        skip = x
        x = F.elu(self.norm1(x))
        x = F.elu(self.norm2(self.conv1(x)))
        x = self.conv2(x) + skip
        return x
    
# We split up our network into two parts, the Encoder and the Decoder
# this is like the encoder part of unet 
class DownBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(DownBlock, self).__init__()
        self.bn1 = nn.GroupNorm(8, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, 2, 1)
        self.bn2 = nn.GroupNorm(8, channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, 1, 1)
        
        self.conv3 = nn.Conv2d(channels_in, channels_out, 3, 2, 1)

    def forward(self, x):
        x = F.elu(self.bn1(x))
                  
        x_skip = self.conv3(x)
        
        x = F.elu(self.bn2(self.conv1(x)))        
        return self.conv2(x) + x_skip
    
# We split up our network into two parts, the Encoder and the Decoder
class UpBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(UpBlock, self).__init__()
        self.bn1 = nn.GroupNorm(8, channels_in)

        self.conv1 = nn.Conv2d(channels_in, channels_in, 3, 1, 1)
        self.bn2 = nn.GroupNorm(8, channels_in)

        self.conv2 = nn.Conv2d(channels_in, channels_out, 3, 1, 1)
        
        self.conv3 = nn.Conv2d(channels_in, channels_out, 3, 1, 1)
        self.up_nn = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x_in):
        x = self.up_nn(F.elu(self.bn1(x_in)))
        
        x_skip = self.conv3(x)
        
        x = F.elu(self.bn2(self.conv1(x)))
        return self.conv2(x) + x_skip


# We split up our network into two parts, the Encoder and the Decoder
class Encoder(nn.Module):
    def __init__(self, channels, ch=32, latent_channels=32):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Conv2d(channels, ch, 3, 1, 1)
        
        self.conv_block1 = DownBlock(ch, ch * 2)
        self.conv_block2 = DownBlock(ch * 2, ch * 4)

        # Instead of flattening (and then having to unflatten) out our feature map and 
        # putting it through a linear layer we can just use a conv layer
        # where the kernal is the same size as the feature map 
        # (in practice it's the same thing)
        self.res_block_1 = ResBlock(ch * 4)
        self.res_block_2 = ResBlock(ch * 4)
        self.res_block_3 = ResBlock(ch * 4)

        self.conv_out = nn.Conv2d(4 * ch, latent_channels, 3, 1, 1)
    
    def forward(self, x):
        x = self.conv_1(x)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = F.elu(self.res_block_3(x))

        return self.conv_out(x)
    

class Decoder(nn.Module):
    def __init__(self, channels, ch = 32, latent_channels = 32):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Conv2d(latent_channels, 4 * ch, 3, 1, 1)
        self.res_block_1 = ResBlock(ch * 4)
        self.res_block_2 = ResBlock(ch * 4)
        self.res_block_3 = ResBlock(ch * 4)

        self.conv_block1 = UpBlock(4 * ch, 2 * ch)
        self.conv_block2 = UpBlock(2 * ch, ch)
        self.conv_out = nn.Conv2d(ch, channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        return torch.tanh(self.conv_out(x))
    

class VQVAE(nn.Module):
    def __init__(self, channel_in, ch=16, latent_channels=32, code_book_dim=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(channels=channel_in, ch=ch, latent_channels=latent_channels)
        
        self.vq = VectorQuantizer(code_book_dim=code_book_dim, 
                                  embedding_dim=latent_channels, 
                                  commitment_cost=commitment_cost)
        
        self.decoder = Decoder(channels=channel_in, ch=ch, latent_channels=latent_channels)

    def encode(self, x):
        encoding = self.encoder(x)
        vq_loss, quantized, encoding_indices = self.vq(encoding)
        return vq_loss, quantized, encoding_indices
        
    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        vq_loss, quantized, encoding_indices = self.encode(x)
        recon = self.decode(quantized)
        
        return recon, vq_loss, quantized
    
dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
random_indices = np.random.choice(len(dataset), size=5000, replace=False)
subset_train_dataset = Subset(dataset, random_indices)
train_loader = DataLoader(dataset=subset_train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.FashionMNIST(root=".data/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# The number of code book embeddings
code_book_dim = 32

# The number of latent embedding channels
latent_channels = 10

# Number of Training epochs
vq_nepoch = 50

# Create our network
vae_net = VQVAE(channel_in=1, latent_channels=latent_channels, ch=16, 
                code_book_dim=code_book_dim, commitment_cost=0.25)

optimizer = torch.optim.Adam(vae_net.parameters(), lr=3e-4)




# Create loss logger
recon_loss_log = []
qv_loss_log = []
test_recon_loss_log = []
train_loss = 0




for epoch in range(5):
    train_loss = 0
    vae_net.train()
    for i, data in enumerate(tqdm(train_loader, leave=False, desc="Training")):

        image = data[0]
            # Forward pass the image in the data tuple
        recon_data, vq_loss, quantized = vae_net(image)

            # Calculate the loss
        recon_loss = (recon_data - image).pow(2).mean()
        loss = vq_loss + recon_loss

        # Take a training step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the loss
        recon_loss_log.append(recon_loss.item())
        qv_loss_log.append(vq_loss.item())
        train_loss += recon_loss.item()
        

    vae_net.eval()
    for i, data in enumerate(tqdm(test_loader, leave=False, desc="Testing")):
        image = data[0]
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                # Forward pass the image in the data tuple
                recon_data, vq_loss, quantized = vae_net(image)

                # Calculate the loss
                recon_loss = (recon_data - image).pow(2).mean()
                loss = vq_loss + recon_loss
                test_recon_loss_log.append(recon_loss.item())

print("ran code!")
