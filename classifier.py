import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network
class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        # Define a simple feedforward architecture
        self.flatten = nn.Flatten()  # flatten 28x28 images into a 784-dimensional vector
        self.fc1 = nn.Linear(784, 128)  # first fully connected layer, 784 to 128
        self.relu1 = nn.ReLU()  # activation function
        self.fc2 = nn.Linear(128, 64)  # second fully connected layer, 128 to 64
        self.relu2 = nn.ReLU()  # activation function
        self.fc3 = nn.Linear(64, 10)  # output layer, 64 to 10 (10 digits)
        self.softmax = nn.LogSoftmax(dim=1)  # output as probabilities

    def forward(self, x):
        x = self.flatten(x)  # flatten input
        x = self.fc1(x)  # first layer
        x = self.relu1(x)  # activation
        x = self.fc2(x)  # second layer
        x = self.relu2(x)  # activation
        x = self.fc3(x)  # output layer
        x = self.softmax(x)  # softmax activation for probabilities
        return x

# Set up the data loading and transformations
transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST datasets
DATA_DIR = "."
download_dataset = True  # set to False if already downloaded

mnist_train = datasets.MNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
mnist_test = datasets.MNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

# Split the training set into training and validation sets
train_size = int(0.8 * len(mnist_train))
val_size = len(mnist_train) - train_size
mnist_training, mnist_validation = data.random_split(mnist_train, [train_size, val_size])

# Create data loaders for batching
batch_size = 64
train_loader = data.DataLoader(mnist_training, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(mnist_validation, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

# Instantiate the model
model = FeedForwardNN()

# Define the loss function and optimizer
loss_function = nn.NLLLoss()  # Negative Log-Likelihood Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Train the model
num_epochs = 10  # number of epochs
model.train()  # set the model in training mode

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()  # reset gradients
        outputs = model(images)  # forward pass
        loss = loss_function(outputs, labels)  # compute loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights
        total_loss += loss.item()  # accumulate loss

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

print("Training complete!")