#Author: Sakasa Lernia
#Student no: SKSLER002
#Date 21 April 2024
#ANN for classifying hand drawn images

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  #Used for plotting training loss and validation accuracy graph
import numpy as np
import os
import random

# Defining the neural network
class ANNReLU(nn.Module):
    def __init__(self):
        super(ANNReLU, self).__init__()

        # Using relu activation fucntion for the hidden layers and logSoftmax for the output layer.
        self.flatten = nn.Flatten()  # Flattening the input
        self.fc1 = nn.Linear(784, 128)  # 1st hidden layer with 128 nodes
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)  #Second HIDDEN LAYER
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)  # Output layer
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = self.fc1(x)  # First hidden layer
        x = self.relu1(x)
        x = self.fc2(x)  # Second hidden layer
        x = self.relu2(x)
        x = self.fc3(x)  # Output
        x = self.log_softmax(x)
        return x

# Function to classify images with the trained model
def classify_image(model, file_path):
    # Load and preprocess the image
    image = Image.open(file_path).convert("L")  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)  # Get model output
        _, predicted = torch.max(output, dim=1)  # Get predicted class
        return predicted.item()


# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

DATA_DIR = "."
mnist_train = datasets.MNIST(DATA_DIR, train=True, download=False, transform=transform)
mnist_test = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)

# Split into training and validation sets
train_size = int(0.8 * len(mnist_train))
val_size = len(mnist_train) - train_size
mnist_training, mnist_validation = data.random_split(mnist_train, [train_size, val_size])

train_loader = data.DataLoader(mnist_training, batch_size=32, shuffle=True)
val_loader = data.DataLoader(mnist_validation, batch_size=32, shuffle=False)

# Initialize the model, optimizer, and loss function
model = ANNReLU()
loss_function = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with a learning rate of 0.001

num_epochs = 25 #previous 80->45->30

# Lists to track metrics
train_losses = []
val_accuracies = []

print("Training the model...")

# Training loop with a fixed number of epochs
for epoch in range(num_epochs):
    model.train()  # Set to training mode
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()  # Resetting gradients
        outputs = model(images)  # Forward pass
        loss = loss_function(outputs, labels)
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        total_loss += loss.item()  # Accumulate total loss

    avg_train_loss = total_loss / len(train_loader)  # Average training loss
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs, dim=1)  # Get predicted labels
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)

    val_accuracy = correct / total  # Calculate validation accuracy
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Training Loss: {avg_train_loss:.4f}, "
          f"Validation Accuracy: {val_accuracy:.4f}")

print("Training completed!")

# Allowing user input to classify an image
import PIL.Image as Image

while True:
    file_path = input("Please enter a filepath: ")

    if file_path.lower() == "exit":  # Exit loop
        print("Exiting...")
        break

    if not os.path.exists(file_path):
        print("The specified file path does not exist. Please try again.")  # invalid paths
    else:
        predicted_class = classify_image(model, file_path)  # Classify using the helper function
        print(f"Classifier: {predicted_class}")
