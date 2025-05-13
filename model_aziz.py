from torch import nn
import torch.nn.functional as F
import torch


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, z_dim=20):
        super().__init__()
        # encoding 
        self.img2hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2mu = nn.Linear(hidden_dim, z_dim)
        self.hidden2sigma = nn.Linear(hidden_dim, z_dim)

        # decoder 
        self.z2hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden2image = nn.Linear(hidden_dim, input_dim)
        
        # define ReLU
        self.relu = nn.ReLU()

    def encoder(self, x):
        h = self.relu(self.img2hidden(x))
        h = self.relu(self.hidden2hidden(h))
        mu = self.hidden2mu(h)
        logvar = self.hidden2sigma(h) 
        return mu, logvar
    
    def decoder(self, z):
        h = self.relu(self.z2hidden(z))
        h = self.relu(self.hidden2hidden(h))
        img = self.hidden2image(h)
        img = torch.sigmoid(img)
        return img


    def forward(self, x):
        mu, logvar = self.encoder(x)
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon

        x_reconstructed = self.decoder(z_reparameterized)

        return x_reconstructed, mu, logvar

class CNN(nn.Module):
    def __init__(self, image_dim=28, num_classes=10):
        super().__init__()
        self.input_len = int(64 * image_dim / 2 * image_dim / 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride = 1, padding=1, padding_mode='reflect')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,padding_mode='reflect')
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.input_len, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.out(x)

        return x
    
# adding the VQ-VAE componenents 
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
        # inputs = inputs.permute(0, 2, 3, 1).contiguous() # make sure this works for us PRINT THE SIZE WE WANT THIS TO BE BxHxWxC 
        # input_shape = inputs.shape

        # flat_input = inputs.view(-1, 1, self.embedding_dim)
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate the distance between each embedding and each codebook vector
        distances = torch.sum((flat_input.unsqueeze(1) - self.embedding.weight.unsqueeze(0)) ** 2, dim=2)


        # Find the closest codebook vector
        encoding_indices = torch.argmin(distances, dim=1)

        # get the quantized vector 
        quantized = self.embedding(encoding_indices)

        # Create loss that pulls encoder embeddings and codebook vector selected
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        q_latent_loss = F.mse_loss(quantized, flat_input.detach())
        #e_latent_loss = F.binary_cross_entropy(quantized.detach(), flat_input)
        #q_latent_loss = F.binary_cross_entropy(quantized, flat_input.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Reconstruct quantized representation using the encoder embeddings to allow for 
        # backpropagation of gradients into encoder
        if self.training:
            quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, channel_in, hidden_dim,latent_channels=32, code_book_dim=64, commitment_cost=0.25):
        super().__init__()
        # we will only define the Vector quantizer 
        self.vq = VectorQuantizer(code_book_dim=code_book_dim, 
                                  embedding_dim=latent_channels, 
                                  commitment_cost=commitment_cost)
        
        self.enc1 = nn.Linear(channel_in, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc3 = nn.Linear(hidden_dim, latent_channels)

        self.dec1 = nn.Linear(latent_channels, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec3 = nn.Linear(hidden_dim, channel_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        encoding = self.relu(self.enc3(x))

        vq_loss, quantized, encoding_indicies = self.vq(encoding)
        return vq_loss, quantized, encoding_indicies
    
    def decode(self, x):
        x = self.relu(self.dec1(x))
        x = self.relu(self.dec2(x))
        img = self.dec3(x)
        img = torch.sigmoid(img)
        return img
    
    def forward(self, x):
        vq_loss, quantized, encoding_indicies = self.encode(x)
        recon = self.decode(quantized)
        return recon, vq_loss, quantized
    
         

class ConditionalGAN(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=512, channel_out=784, num_classes=10):  # Increased hidden_dim
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding layer for labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Generator - even deeper with residual connections
        self.generator = nn.Sequential(
            # Initial projection
            nn.Linear(latent_dim + num_classes, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            # Residual block 1
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            # Residual block 2
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            # Upscaling
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            
            # Final layer with pixel activation
            nn.Linear(hidden_dim, channel_out),
            nn.Sigmoid()
        )

        # Discriminator - spectral normalization for stability
        self.discriminator = nn.Sequential(
            # Feature extraction
            nn.utils.spectral_norm(nn.Linear(channel_out + num_classes, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Self-attention layer
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Feature compression
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Classification
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def generator_forward(self, z, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        # Concatenate noise and labels
        z = torch.cat([z, label_embedding], dim=1)
        return self.generator(z)

    def discriminator_forward(self, x, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        # Concatenate images and labels
        x = torch.cat([x, label_embedding], dim=1)
        return self.discriminator(x)

    def generate(self, num_samples, labels, device):
        """Generate images for specific labels"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.generator_forward(z, labels)

class BigGAN(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=512, num_classes=10, ch=64):
        super(BigGAN, self).__init__()
        self.latent_dim = latent_dim
        
        # Generator
        self.generator = nn.Sequential(
            # Initial dense layer
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            # Reshape to start convolutions
            # For 28x28 output, we start with 7x7
            nn.Linear(hidden_dim, 7 * 7 * ch * 8),
            nn.BatchNorm1d(7 * 7 * ch * 8),
            nn.ReLU(),
            
            # Will be reshaped to (batch_size, ch*8, 7, 7)
        )
        
        self.generator_conv = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(ch * 4, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # 14x14 -> 7x7
            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2),
            
            # 7x7 -> 1x1
            nn.Conv2d(ch * 2, 1, 7, stride=1, padding=0),
            nn.Sigmoid()
        )

    def generator_forward(self, z, labels):
        # One-hot encode labels
        labels_onehot = torch.zeros(labels.size(0), 10, device=labels.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Concatenate noise and labels
        z = torch.cat([z, labels_onehot], dim=1)
        
        # Forward through dense layers
        h = self.generator(z)
        
        # Reshape for convolutions: (batch_size, ch*8, 7, 7)
        h = h.view(h.size(0), -1, 7, 7)
        
        # Forward through conv layers
        return self.generator_conv(h)

    def discriminator_forward(self, x, labels):
        # One-hot encode labels
        labels_onehot = torch.zeros(labels.size(0), 10, device=labels.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Process the image through discriminator
        x = self.discriminator(x)
        
        # Combine with labels (you might want to adjust this part based on your needs)
        x = x.view(x.size(0), -1)  # Flatten
        return x.squeeze()

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=False, downsample=False):
        super().__init__()
        self.upsample = upsample
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        h = self.activation(self.bn1(x))
        
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
            
        return h + self.skip(x)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Compute attention scores
        q = self.query(x).view(B, -1, H*W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H*W)
        v = self.value(x).view(B, -1, H*W)
        
        attention = F.softmax(torch.bmm(q, k), dim=2)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x

if __name__ == "__main__":
    print("Testing VAE")
    x = torch.randn(4, 28*28)
    vae = VariationalAutoEncoder(input_dim=784)
    x_, mu, sigma = vae(x)
    print(x_.shape)
    print(mu.shape)
    print(sigma.shape)

    print("Testing CNN")
    x2 = torch.randn(4,1,28,28)
    cnn = CNN()
    output = cnn(x2)
    print(output.shape)

    print("Testing VQ-VAE")
    x = torch.randn(4, 28*28)
    vqvae = VQVAE(channel_in=784, hidden_dim=10, latent_channels=4, code_book_dim=4)
    recon, vq_loss, quantized = vqvae(x)
    print("Finished running the VQ-VAE")
    print("SHAPE OF THE RECON: ", recon.shape)
    print("The loss", vq_loss)
    print("The quantized representation", quantized.shape)

    print("\n=== Model Parameter Counts ===")
    
    models = {
        'VAE': VariationalAutoEncoder(input_dim=784),
        'VQ-VAE': VQVAE(channel_in=784, hidden_dim=10, latent_channels=4, code_book_dim=4),
        'Conditional GAN': ConditionalGAN(latent_dim=100, hidden_dim=512, channel_out=784, num_classes=10),
        'BigGAN': BigGAN(latent_dim=128, hidden_dim=512, num_classes=10, ch=64),
        'CNN': CNN(image_dim=28, num_classes=10)
    }
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{name}:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")





