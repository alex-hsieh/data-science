from medmnist import DermaMNIST
import numpy as np
import matplotlib.pyplot as plt

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

"""
# Verifying the shapes of the datasets
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)
"""

# Find the unique classes and count how many times each appears
classes, counts = np.unique(y_train, return_counts=True)

# Normalize the pixel values to be between 0.0 and 1.0
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