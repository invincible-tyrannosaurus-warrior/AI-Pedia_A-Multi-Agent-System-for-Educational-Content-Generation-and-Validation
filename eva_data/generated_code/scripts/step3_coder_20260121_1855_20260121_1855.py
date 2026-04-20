import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the path for saving assets
ASSETS_PATH = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'

# Check if the assets directory exists, create it if not
os.makedirs(ASSETS_PATH, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
image_size = 64
num_channels = 3
latent_dim = 100
num_epochs = 50
learning_rate = 0.0002
beta1 = 0.5

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(
    root='D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code',
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, num_channels, features_g=64):
        super(Generator, self).__init__()
        # Input is Z, going into a convolution
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            # state size. (features_g*8) x 4 x 4
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            # state size. (features_g*4) x 8 x 8
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            # state size. (features_g*2) x 16 x 16
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            # state size. (features_g) x 32 x 32
            nn.ConvTranspose2d(features_g, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, num_channels, features_d=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (num_channels) x 64 x 64
            nn.Conv2d(num_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d) x 32 x 32
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*2) x 16 x 16
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*4) x 8 x 8
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*8) x 4 x 4
            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

# Initialize networks
netG = Generator(latent_dim, num_channels).to(device)
netD = Discriminator(num_channels).to(device)

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# Initialize optimizers
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Lists to store losses for plotting
D_losses = []
G_losses = []
iters = []

# Training loop
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), 1, dtype=torch.float, device=device)
        
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(0)
        
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        
        # Update G
        optimizerG.step()

        # Save losses for plotting
        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            
            D_losses.append(errD.item())
            G_losses.append(errG.item())
            iters.append(i)

        # Save generated images every 500 iterations
        if i % 500 == 0:
            with torch.no_grad():
                fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
                fake = netG(fixed_noise).detach().cpu()
            
            # Create grid of images
            fig = plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title("Generated Images")
            plt.imshow(np.transpose(torchvision.utils.make_grid(
                fake, padding=2, normalize=True), (1, 2, 0)))
            
            # Save the plot
            plt.savefig(os.path.join(ASSETS_PATH, f'generated_images_epoch_{epoch}_iter_{i}.png'))
            plt.close(fig)

# Plot training losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(D_losses, label="D")
plt.plot(G_losses, label="G")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(ASSETS_PATH, 'losses.png'))
plt.close()

# Save final generated images
with torch.no_grad():
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    fake = netG(fixed_noise).detach().cpu()

# Create and save final grid of generated images
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Final Generated Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(
    fake, padding=2, normalize=True), (1, 2, 0)))
plt.savefig(os.path.join(ASSETS_PATH, 'final_generated_images.png'))
plt.close(fig)

print("Training completed!")