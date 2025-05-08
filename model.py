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




if __name__ == "__main__":
    x = torch.randn(4, 28*28)
    vae = VariationalAutoEncoder(input_dim=784)
    x_, mu, sigma = vae(x)
    print(x_.shape)
    print(mu.shape)
    print(sigma.shape)