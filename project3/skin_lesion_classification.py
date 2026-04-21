"""
Project 3: Skin Lesion Classification
Author: Alexander Hsieh
Description: This script loads the DermaMNIST dataset, explores class distributions,
and trains both a baseline Logistic Regression model and a Neural Network (MLP)
to classify 7 types of skin lesions.
"""

from medmnist import DermaMNIST
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

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

# Print dataset sizes and dimensions
print(f"Train: {X_train.shape}, Labels: {y_train.shape}")
print(f"Val:   {X_val.shape}, Labels: {y_val.shape}")
print(f"Test:  {X_test.shape}, Labels: {y_test.shape}")


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

# Display one sample image per class
fig, axes = plt.subplots(1, 7, figsize=(14, 2))
for cls in range(7):
    idx = np.where(y_train.ravel() == cls)[0][0]
    axes[cls].imshow(X_train[idx])
    axes[cls].set_title(f"Class {cls}")
    axes[cls].axis('off')
plt.suptitle("Sample Images per Class")
plt.show()

# --- PART 2: Logistic Regression ---

# reshaping the data to be 2D (num_samples, num_features)
# Scikit-Learn models cannot accept 4D image arrays. We use .reshape() to flatten the 28x28x3 images into a single 1D vector of 2,352 features for each patient. The '-1' tells NumPy to calculate the remaining dimension automatically.
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Tune C hyperparameter by trying multiple values
C_values = [0.01, 0.1, 1.0, 10.0]
best_C = None
best_val_acc = 0

for C in C_values:
    model = LogisticRegression(C=C, max_iter=3000, random_state=42)
    model.fit(X_train_reshaped, y_train.ravel())
    acc = accuracy_score(y_val, model.predict(X_val_reshaped))
    print(f"C={C}: Val Accuracy = {acc:.4f}")
    if acc > best_val_acc:
        best_val_acc = acc
        best_C = C

# Train final model with best C, and track training time
print(f"\nBest C: {best_C}")
start = time.time()
logistic_model = LogisticRegression(C=best_C, max_iter=3000, random_state=42, class_weight='balanced')
logistic_model.fit(X_train_reshaped, y_train.ravel())
lr_train_time = time.time() - start
print(f"Logistic Regression Training Time: {lr_train_time:.2f}s")

# Make predictions on the validation and test sets
y_val_pred = logistic_model.predict(X_val_reshaped)
y_test_pred = logistic_model.predict(X_test_reshaped)

# Evaluate the model using accuracy score
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Classification Report on Validation Set:")
print(classification_report(y_val, y_val_pred))
print("Classification Report on Test Set:")
print(classification_report(y_test, y_test_pred))

# Display the Confusion Matrix for Logistic Regression
ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# --- PART 3: Building a Neural Network ---
# We initialize a Multi-Layer Perceptron.
# Architecture: Two hidden layers with 128 and 64 neurons respectively.
# Activation: 'relu' activation function helps the network learn non-linear patterns.
# Loss: log_loss (cross-entropy). Optimizer: adam.
nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=3000, random_state=42)

# Track training time
start = time.time()
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train.ravel())
nn_model.fit(X_train_reshaped, y_train.ravel(), sample_weight=sample_weights)
nn_train_time = time.time() - start
print(f"Neural Network Training Time: {nn_train_time:.2f}s")

# Predictions on validation and test sets
y_val_pred_nn = nn_model.predict(X_val_reshaped)
y_test_pred_nn = nn_model.predict(X_test_reshaped)

print("Neural Network Classification Report on Validation Set:")
print(classification_report(y_val, y_val_pred_nn, zero_division=0))

print("Neural Network Classification Report on Test Set:")
print(classification_report(y_test, y_test_pred_nn, zero_division=0))

# Confusion matrices
ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred_nn)
plt.title("Neural Network Confusion Matrix (Validation)")
plt.show()

# Plot training loss curve
plt.plot(nn_model.loss_curve_)
plt.title("Neural Network Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# --- PART 4: Model Comparison ---


lr_acc = accuracy_score(y_test, y_test_pred)
nn_acc = accuracy_score(y_test, y_test_pred_nn)

lr_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
nn_precision = precision_score(y_test, y_test_pred_nn, average='weighted', zero_division=0)

lr_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
nn_recall = recall_score(y_test, y_test_pred_nn, average='weighted', zero_division=0)

lr_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
nn_f1 = f1_score(y_test, y_test_pred_nn, average='weighted', zero_division=0)

comparison = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Training Time (s)"],
    "Logistic Regression": [lr_acc, lr_precision, lr_recall, lr_f1, round(lr_train_time, 2)],
    "Neural Network":      [nn_acc, nn_precision, nn_recall, nn_f1, round(nn_train_time, 2)]
}


df = pd.DataFrame(comparison)
print(df.to_string(index=False))