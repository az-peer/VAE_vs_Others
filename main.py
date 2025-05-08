import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from model import VariationalAutoEncoder
import numpy as np
import argparse
from engine import train_model, inference

###################################### PARSING ARGUMENTS ######################################
parser = argparse.ArgumentParser(description='Train VAE model')
parser.add_argument('--input_dim', type=int, default=784, help='Input dimension')
parser.add_argument('--hidden_dim', type=int, default=200, help='Hidden dimension')
parser.add_argument('--z_dim', type=int, default=20, help='Latent dimension')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (Karpathy constant)')
parser.add_argument('--add_noise', action='store_true', help='Add noise to inputs')
parser.add_argument('--noise_factor', type=float, default=0.3, help='Noise factor')
parser.add_argument('--model_save_path', type=str, default='vae_model.pth', help='Path to save the model')
parser.add_argument('--inference_save_path', type=str, default='inference_reconstruction.png', help='Path to save the inference image')
parser.add_argument('--num_samples', type=int, default=8, help='Number of samples for inference')
parser.add_argument('--dataset_save_path', type=str, default='generated_dataset.pt', help='Path to save the generated dataset')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = args.input_dim
hidden_dim = args.hidden_dim
z_dim = args.z_dim
num_epochs = args.num_epochs
batch_size = args.batch_size
karpathy_constant = args.lr
add_noise_flag = args.add_noise
noise_factor = args.noise_factor
model_save_path = args.model_save_path
inference_save_path = args.inference_save_path
num_samples = args.num_samples  
dataset_save_path = args.dataset_save_path

###################################### LOADING DATASET ######################################
dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
random_indices = np.random.choice(len(dataset), size=5000, replace=False)
subset_train_dataset = Subset(dataset, random_indices)
train_loader = DataLoader(dataset=subset_train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.FashionMNIST(root=".data/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

###################################### EXPERIMENT SETUP ######################################
model = VariationalAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=karpathy_constant)
loss_fn = nn.BCELoss(reduction='sum')

###################################### TRAINING ######################################
train_model(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    num_epochs=num_epochs,
    input_dim=input_dim,
    add_noise_flag=add_noise_flag,
    noise_factor=noise_factor,
    save_path=model_save_path
)

###################################### INFERENCE ######################################
inference(
    model=model,
    test_loader=test_loader,
    device=device,
    num_samples=num_samples,
    save_path=inference_save_path,
    dataset_save_path=dataset_save_path
)
