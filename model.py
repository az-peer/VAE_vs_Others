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


