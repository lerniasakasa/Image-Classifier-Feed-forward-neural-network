##Author Information
Author: Sakasa Lernia
Date: April 21, 2024

## Overview
This program is for a simple Artificial Neural Network (ANN) that classifies hand-drawn images using the MNIST dataset, a benchmark dataset containing images of handwritten digits from 0 to 9. The model is trained and evaluated on this dataset and includes a method for classifying new images based on the trained model.

## Requirements
To run the code, you need the following dependencies installed:

Python 3.7 or later
PyTorch
torchvision
matplotlib
PIL (Python Imaging Library)

## Files and Structure

ANNReLU: This class defines the neural network architecture using PyTorch.
classify_image(model, file_path): This function classifies a single hand-drawn image based on a trained model.
train_loader, val_loader: DataLoaders for training and validation datasets, respectively.
model, loss_function, optimizer: Initialization of the ANN, loss function, and optimizer for training.
Training loop: The main training loop for the ANN with validation steps and tracking of metrics.
Interactive Image Classification: A small interactive section to classify user-specified images with the trained model.

## Training the Model
You need to have the MNIST Data Set downloaded. The training loop runs for 25 epochs, allowing the model to learn from the training data and evaluate its performance on the validation set. During training, the loss function and optimizer are used to adjust the model's parameters to improve accuracy.

To run and use start the model, run:
	python classifier.py
	
After training, you can use the classify_image function to classify hand-drawn images. The script provides an interactive loop that prompts you to enter a file path to classify an image. To exit the loop, type "exit".
!!!Ensure the image file is in a compatible format. The script will provide the predicted class for the given image.

## Usage Notes
Data Directory: The dataset is expected to be in the current directory (DATA_DIR = "."). Adjust the DATA_DIR variable if your dataset is in a different location.
