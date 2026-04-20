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

# --- PART 2: Logistic Regression ---
# reshaping the data to be 2D (num_samples, num_features)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

from sklearn.linear_model import LogisticRegression

# Create and fit the logistic regression model using .fit()
logistic_model = LogisticRegression(max_iter=3000)
logistic_model.fit(X_train_reshaped, y_train.ravel())

# Make predictions on the validation set
y_val_pred = logistic_model.predict(X_val_reshaped)
y_test_pred = logistic_model.predict(X_test_reshaped)

# Evaluate the model using accuracy score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Classification Report on Validation Set:")
print (classification_report(y_val, y_val_pred))

# --- PART 3: Building a Neural Network ---
from sklearn.neural_network import MLPClassifier
nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=3000)

# Fit the neural network model using .fit()
nn_model.fit(X_train_reshaped, y_train.ravel()) 
y_val_pred_nn = nn_model.predict(X_val_reshaped)
print (classification_report(y_val, y_val_pred_nn))