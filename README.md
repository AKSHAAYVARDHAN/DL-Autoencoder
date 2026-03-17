# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

This code implements a Denoising Autoencoder using PyTorch to clean noisy images from the MNIST dataset. It uses a convolutional neural network architecture, where the encoder compresses the input image into a lower-dimensional representation, and the decoder reconstructs the original image from this compressed form. To train the model to remove noise, Gaussian noise is added to the clean images, and the network learns to recover the original from the noisy version. The training process uses Mean Squared Error (MSE) as the loss function to measure the reconstruction error and the Adam optimizer to update the model weights. The autoencoder is trained over multiple epochs using mini-batches of data for efficiency. After training, the model's performance is visually evaluated by displaying the original, noisy, and denoised images side by side.

## DESIGN STEPS
### STEP 1: 
Examine the current setup for loading the MNIST dataset and the add_noise function to understand how input data is prepared and corrupted. This includes checking the transform applied to images.



### STEP 2: 
Review the DenoisingAutoencoder class definition, paying attention to the encoder and decoder layers, and their corresponding activation functions. Also, confirm the criterion (loss function) and optimizer used for training.



### STEP 3: 
Examine the train function to understand how the model is trained, including the epoch loop, batch processing, noise addition during training, forward pass, loss calculation, backpropagation, and optimizer step.


### STEP 4: 

Review the visualize_denoising function to understand how the model's performance is evaluated and visualized. Pay attention to how original, noisy, and denoised images are displayed side-by-side for comparison.

### STEP 5: 
Based on the output of the executed visualize_denoising function, analyze the effectiveness of the current autoencoder in removing noise. Identify patterns in denoised images and consider if the model is underfitting or overfitting.


### STEP 6: 
Summarize the current understanding of the denoising autoencoder's implementation and performance, and suggest potential next steps for improvement or further analysis based on the assessment.

## PROGRAM

### Name: Akshaay Vardhan S

### Register Number: 212224220007

```python
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('Name: Akshaay Vardhan S')
print('Register Number: 212224220007')
summary(model, input_size=(1, 28, 28))

def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print('Name: Akshaay Vardhan S')
    print('Register Number: 212224220007')
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

def visualize_denoising(model, loader, num_images=10):
    model.eval()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print('Name: Akshaay Vardhan S')
    print('Register Number: 212224220007')
    plt.figure(figsize=(18, 6))

    for i in range(num_images):
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")


        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")


        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)

```

### OUTPUT

### Model Summary
<img width="648" height="502" alt="image" src="https://github.com/user-attachments/assets/2bdd95cd-66ee-4758-9145-3cda7b4b71dd" />


### Training loss

<img width="306" height="155" alt="image" src="https://github.com/user-attachments/assets/bb8be1ea-81df-4f01-b7b0-35d64f440fe4" />


## Original vs Noisy Vs Reconstructed Image
<img width="1691" height="604" alt="image" src="https://github.com/user-attachments/assets/596e5e2e-21e4-4a6b-a276-35125c3770df" />


## RESULT
Thus, develop a convolutional autoencoder for image denoising application excuted succesfully


