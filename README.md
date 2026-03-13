# MNIST Handwritten Digit Classifier
A simple neural network built with PyTorch to classify handwritten digits from the MNIST dataset.
## Project Overview
This project implements a feedforward neural network with:
- Input layer: 784 neurons (28x28 pixel images)
- Hidden layers: 128 and 64 neurons with ReLU activation
- Output layer: 10 neurons (digits 0-9)
## Technologies Used
- Python
- PyTorch
- torchvision
## Model Architecture
Input (784) → FC (128) → ReLU → FC (64) → ReLU → FC (10) → Output
## Training
The model is trained for 3 epochs using:
- Loss function: Cross-Entropy Loss
- Optimizer: Adam (learning rate = 0.001)
- Batch size: 64
## Results
The model achieves competitive accuracy on MNIST digit classification after just 3 epochs of training.
## How to Run
```bash
python train.py```
The script will automatically download the MNIST dataset and train the model.
