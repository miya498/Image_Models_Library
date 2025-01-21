import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from unet import UNet
from diffusion_model import create_diffusion_model
from training import train_model
from sampling import sample_images

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
img_size = 28
num_timesteps = 1000
epochs = 10
lr = 1e-3

# Dataset and Dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model and Diffusion Setup
model = UNet(in_ch=1, out_ch=1)
diffusion = create_diffusion_model(num_timesteps)

# Training
losses = train_model(model, diffusion, dataloader, epochs, num_timesteps, lr, device)

# Sampling
sample_images(diffusion, model, num_samples=25, img_size=img_size, num_timesteps=num_timesteps, device=device)