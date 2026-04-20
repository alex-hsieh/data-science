"""
Project 3: Skin Lesion Classification
Author: Alexander Hsieh
Description: This script loads the DermaMNIST dataset, explores class distributions, 
and trains both a baseline Logistic Regression model and a Neural Network (MLP) 
to classify 7 types of skin lesions.
"""

from medmnist import DermaMNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# --- PART 1: Data Loading and Preprocessing ---

# Extracting the images and labels from the dataset
train_dataset = DermaMNIST(split="train", download=True)

# Getting the variables to be used for training the model
X_train = train_dataset.imgs
y_train = train_dataset.labels

# Getting the variables for validating the model
val_dataset = DermaMNIST(split="val", download=True)
X_val = val_dataset.imgs
y_val = val_dataset.labels

# Getting the varables for testing the model
test_dataset = DermaMNIST(split="test", download=True)
X_test = test_dataset.imgs
y_test = test_dataset.labels

# Find the unique classes and count how many times each appears
classes, counts = np.unique(y_train, return_counts=True)

# Normalize the pixel values to be between 0.0 and 1.0
# Scaling pixel values from (0-255) to (0.0-1.0) is crucial for machine learning models. It helps the math (gradient descent) converge much faster and prevents larger pixel values from dominating the learning process.
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Let's plot it as a bar chart so you can easily take a screenshot for your report!
plt.bar(classes, counts)
plt.title("Class Distribution in Training Set")
plt.xlabel("Class Label")
plt.ylabel("Number of Images")
plt.show()

# Display the very first image in the training set
plt.imshow(X_train[0])
plt.title(f"Class Label: {y_train[0][0]}")
plt.show()

# --- PART 2: Logistic Regression ---

# reshaping the data to be 2D (num_samples, num_features)
# Scikit-Learn models cannot accept 4D image arrays. We use .reshape() to flatten the 28x28x3 images into a single 1D vector of 2,352 features for each patient. The '-1' tells NumPy to calculate the remaining dimension automatically.
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)



# Create and fit the logistic regression model using .fit()
# We set max_iter=3000 to give the optimization algorithm enough time to converge, as the high-dimensional image data makes finding a solution difficult.
logistic_model = LogisticRegression(max_iter=3000)

# We use .ravel() on the y_train labels to convert the 2D column vector into a 1D array, which prevents scikit-learn from throwing DataConversionWarnings.
logistic_model.fit(X_train_reshaped, y_train.ravel())

# Make predictions on the validation set
y_val_pred = logistic_model.predict(X_val_reshaped)
y_test_pred = logistic_model.predict(X_test_reshaped)

# Evaluate the model using accuracy score
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Classification Report on Validation Set:")

# The classification report outputs Precision, Recall, and F1-score, which are critical for evaluating model performance on this highly imbalanced dataset.
print (classification_report(y_val, y_val_pred))

# Display the Confusion Matrix for Logistic Regression
ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# --- PART 3: Building a Neural Network ---
# We initialize a Multi-Layer Perceptron. 
# Architecture: Two hidden layers with 128 and 64 neurons respectively.
# Activation: It uses the default 'relu' activation function, which helps the network learn non-linear patterns.
nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=3000)

# Fit the neural network model using .fit()
nn_model.fit(X_train_reshaped, y_train.ravel()) 
y_val_pred_nn = nn_model.predict(X_val_reshaped)
print("Neural Network Classification Report on Validation Set:")
print (classification_report(y_val, y_val_pred_nn, zero_division=0))  # zero_division=0 prevents warnings about undefined metrics when a class has no predicted samples

# Display the Confusion Matrix for the Neural Network
ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred_nn)
plt.title("Neural Network Confusion Matrix")
plt.show()