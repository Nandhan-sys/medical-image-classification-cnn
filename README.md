Medical Image Classification using CNN

A deep learning project using PyTorch to classify chest X-ray images into two classes:

NORMAL

PNEUMONIA

Project Overview

The goal of this project is to build a Convolutional Neural Network (CNN) that can classify chest X-ray images as NORMAL or PNEUMONIA.

Technologies

Python

PyTorch

Torchvision

Scikit-learn

Matplotlib

Seaborn

Google Colab

CUDA / Tesla T4 GPU

Dataset

Dataset: Chest X-Ray Pneumonia dataset from Kaggle.

The dataset contains chest X-ray images organized into NORMAL and PNEUMONIA classes.

A stratified 90/10 split was created from the original training data to obtain a larger validation set.

The original test set was kept separate and used only for the final evaluation.

Data Preprocessing

Training images:

Resize → Random Horizontal Flip → ToTensor → Normalize

Validation and test images:

Resize → ToTensor → Normalize

Images were resized to 224 × 224 pixels.

Model

A custom CNN was built using PyTorch.

Architecture:

Input Image
    ↓
Conv2D (3 → 10)
    ↓
ReLU
    ↓
MaxPooling
    ↓
Conv2D (10 → 20)
    ↓
ReLU
    ↓
Flatten
    ↓
Fully Connected Layer
    ↓
2 Classes


Output classes:

0 → NORMAL
1 → PNEUMONIA

Class Imbalance

The training dataset contained more PNEUMONIA images than NORMAL images.

To study the effect of class imbalance, two CNN experiments were performed:

CNN without class weights

CNN with weighted CrossEntropyLoss

Validation Results
Model	Accuracy	Precision	Recall	F1
CNN without weights	97.32%	98.20%	98.20%	98.20%
Weighted CNN	96.55%	97.93%	97.42%	97.67%

The unweighted CNN performed better on this validation split, while the weighted model was useful for studying the effect of class imbalance.

Final Test Results

The final unweighted CNN was evaluated on the locked test set.

Metric	Result
Accuracy	72.76%
Precision	69.86%
Recall	99.23%
F1-score	81.99%
Confusion Matrix
                 Predicted
              NORMAL  PNEUMONIA

Actual NORMAL    67       167
Actual PNEUMONIA  3       387


The model detected most PNEUMONIA cases, but it also incorrectly classified a number of NORMAL images as PNEUMONIA.

Learning Outcomes

Through this project, I learned:

Image preprocessing using PyTorch

PyTorch Dataset and DataLoader

Building a CNN using nn.Module

Forward propagation

CrossEntropyLoss

Adam optimizer

Backpropagation

Training and evaluation modes

Handling class imbalance with class weights

Confusion matrix

Accuracy, precision, recall and F1-score

GPU training using CUDA

Limitations

This project is an educational deep learning project and is not intended for clinical diagnosis.

The test results also show that the model produces a relatively high number of false-positive PNEUMONIA predictions.

Project Structure
medical-image-classification-cnn/
│
├── Medical_Image_Classification.ipynb
├── README.md
└── .gitignore
