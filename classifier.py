import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

# Define a different neural network
class DeepFeedForwardNN(nn.Module):
    def __init__(self):
        super(DeepFeedForwardNN, self).__init__()
        self.flatten = nn.Flatten()  # flatten 28x28 images into a 784-dimensional vector
        self.fc1 = nn.Linear(784, 256)  # first fully connected layer, 784 to 256
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)  # activation function with a slight negative slope
        self.fc2 = nn.Linear(256, 512)  # second fully connected layer, 256 to 512
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)  # activation function
        self.fc3 = nn.Linear(512, 256)  # third fully connected layer, 512 to 256
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.01)  # activation function
        self.fc4 = nn.Linear(256, 10)  # output layer, 256 to 10 (10 digits)
        self.softmax = nn.LogSoftmax(dim=1)  # output as probabilities

    def forward(self, x):
        x = self.flatten(x)  # flatten input
        x = self.fc1(x)  # first layer
        x = self.leaky_relu1(x)  # activation
        x = self.fc2(x)  # second layer
        x = self.leaky_relu2(x)  # activation
        x = self.fc3(x)  # third layer
        x = self.leaky_relu3(x)  # activation
        x = self.fc4(x)  # output layer
        x = self.softmax(x)  # softmax activation for probabilities
        return x


# Load MNIST datasets
transform = transforms.Compose([transforms.ToTensor()])

DATA_DIR = "."
mnist_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

train_size = int(0.8 * len(mnist_train))
val_size = len(mnist_train) - train_size
mnist_training, mnist_validation = data.random_split(mnist_train, [train_size, val_size])

train_loader = data.DataLoader(mnist_training, batch_size=32, shuffle=True)  # smaller batch size
val_loader = data.DataLoader(mnist_validation, batch_size=32, shuffle=False)
test_loader = data.DataLoader(mnist_test, batch_size=32, shuffle=False)

# Instantiate the model and set the training configuration
model = DeepFeedForwardNN()
loss_function = nn.CrossEntropyLoss()  # loss function for classification
optimizer = optim.Adagrad(model.parameters(), lr=0.005)  # different optimizer with smaller learning rate

# Early stopping based on validation accuracy
early_stop_patience = 5
best_val_accuracy = 0
epochs_since_last_improvement = 0

# Number of epochs
num_epochs = 40

# Lists to store loss and accuracy for plotting
train_losses = []
val_accuracies = []

# Training loop with early stopping
for epoch in range(num_epochs):
    if epochs_since_last_improvement >= early_stop_patience:
        print(f"Early stopping at epoch {epoch}")
        break

    model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()  # reset gradients
        outputs = model(images)  # forward pass
        loss = loss_function(outputs, labels)  # compute loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights
        total_loss += loss.item()  # accumulate loss

    avg_train_loss = total_loss / len(train_loader)  # average training loss
    train_losses.append(avg_train_loss)

    # Validation accuracy calculation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)  # forward pass
            _, predicted = torch.max(outputs, dim=1)  # get predicted labels
            correct += (predicted == labels).sum().item()  # count correct predictions
            total += labels.size(0)

    val_accuracy = correct / total  # calculate accuracy
    val_accuracies.append(val_accuracy)

    # Check for early stopping condition based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_since_last_improvement = 0
    else:
        epochs_since_last_improvement += 1

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

print("Training complete!")

# Plot training loss and validation accuracy

epochs = list(range(1, len(train_losses) + 1))
fig, ax1 = plt.subplots()

ax1.set_xlabel("Epochs")
ax1.set_ylabel("Training Loss", color='tab:red')
ax1.plot(epochs, train_losses, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel("Validation Accuracy", color='tab:blue')
ax2.plot(epochs, val_accuracies, color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title("Training Loss and Validation Accuracy")
plt.show()

#test accuracy
# Pick 5 random images from the test set
import random

# Get 5 random indices for test data
random_indices = random.sample(range(len(mnist_test)), 5)

# Get the corresponding images and labels
images, labels = zip(*[mnist_test[i] for i in random_indices])

# Make predictions on these images
model.eval()  # set the model to evaluation mode
with torch.no_grad():  # no gradient calculation
    predictions = [model(image.unsqueeze(0)) for image in images]  # get model output for each image
    predicted_labels = [torch.argmax(prediction, dim=1).item() for prediction in predictions]  # get the predicted labels

# Print the predictions and check accuracy
for idx, (image, label, predicted_label) in enumerate(zip(images, labels, predicted_labels)):
    accuracy = predicted_label == label
    print(f"Image {idx+1}: Predicted = {predicted_label}, True Label = {label}, Correct = {accuracy}")

    # Display the image
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title(f"Predicted: {predicted_label}, True Label: {label}")
    plt.show()
