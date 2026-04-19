from medmnist import DermaMNIST
import test

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
