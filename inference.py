import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from model import VariationalAutoEncoder


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
input_dim = 28*28
hidden_dim = 200
z_dim = 20

# Load the trained model
model = VariationalAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim).to(device)
model.load_state_dict(torch.load("vae_model.pth")) 
model.eval()

# Load MNIST test data
test_dataset = datasets.FashionMNIST(root=".data/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define inference function
def inference(model, test_loader, device, num_samples=8, save_path="inference_reconstruction.png", dataset_save_path="generated_dataset.pt"):
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

# Call the inference function
if __name__ == "__main__":
    inference(model, test_loader, device, num_samples=8, save_path="inference_reconstruction.png", dataset_save_path="generated_dataset.pt")
