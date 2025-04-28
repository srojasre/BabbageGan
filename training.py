import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import model  

# Configuración 
BATCH_SIZE = 64
NOISE_DIM = 128
IMG_CHANNELS = 3
IMG_SIZE = 32
IMG_DIM = IMG_CHANNELS * IMG_SIZE * IMG_SIZE
EPOCHS = 10  

# Preprocesamiento y DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  
])

dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modelos, optimizadores y función de pérdida
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = model.Generator(NOISE_DIM, IMG_DIM).to(device)
D = model.Discriminator(IMG_DIM).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

criterion = nn.BCELoss()

# Entrenamiento
for epoch in range(EPOCHS):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.view(-1, IMG_DIM).to(device)
        batch_size = real_imgs.size(0)
        
        # Entrena Discriminador
        noise = torch.randn(batch_size, NOISE_DIM).to(device)
        fake_imgs = G(noise)

        D_real = D(real_imgs)
        D_fake = D(fake_imgs.detach())
        real_labels = torch.ones_like(D_real)
        fake_labels = torch.zeros_like(D_fake)
        loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Entrena Generador
        D_fake = D(fake_imgs)
        loss_G = criterion(D_fake, torch.ones_like(D_fake))
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        
        # Muestra avance
        if i % 500 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {i}/{len(dataloader)} \
                  Loss D: {loss_D.item():.4f}, loss G: {loss_G.item():.4f}")

# Visualización de imágenes generadas del generador
def show_images(images, n):
    images = images.view(-1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    images = images[:n].cpu().detach().numpy()
    fig, axes = plt.subplots(1, n, figsize=(n*2, 2))
    for i in range(n):
        axes[i].imshow(np.transpose((images[i] + 1) / 2, (1, 2, 0)))
        axes[i].axis('off')
    plt.show()

# Generar 5 imágenes de ejemplo
noise = torch.randn(5, NOISE_DIM).to(device)
fake_imgs = G(noise)
show_images(fake_imgs, 5)