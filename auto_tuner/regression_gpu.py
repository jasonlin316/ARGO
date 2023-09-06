import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import csv

# Read the CSV file for training data
data = pd.read_csv("aug24_train.csv", header=None)  # Replace "train_data.csv" with the path to your training CSV file

# Drop rows with missing data (less than 8 columns)
data = data.dropna(subset=data.columns[1:], how='any')
data = data[data.iloc[:, 6] != 1] # drop rows if process number = 1

# Convert data to PyTorch tensors and move them to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Separate the target variable (y) and the input features (X)
y_data = data.iloc[:, 0].values
X_data = data.iloc[:, 1:].values

# Normalize the training data (subtract mean and divide by standard deviation)
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

# Convert data to PyTorch tensors
X_data_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
y_data_tensor = torch.tensor(y_data, dtype=torch.float32).to(device)

# Define the multi-layer perceptron model with ReLU activation for non-negativity
class MLPRegression(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegression, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)  # 6 input features and 64 hidden units
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)         # 64 hidden units and 1 output (scalar)

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # ReLU activation for non-negativity
        x = torch.relu(self.layer2(x))
        return torch.relu(self.layer3(x))  # ReLU activation for non-negativity

# Define k-fold cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Training the model with k-fold cross-validation
batch_size = 128 #64
num_epochs = 2000
clip_value = 1.0

models = []  # List to store trained models for each fold

for fold, (train_indices, val_indices) in enumerate(kf.split(X_data_tensor)):
    X_train_fold = X_data_tensor[train_indices]
    y_train_fold = y_data_tensor[train_indices]
    X_val_fold = X_data_tensor[val_indices]
    y_val_fold = y_data_tensor[val_indices]

    input_dim = X_train_fold.shape[1]
    model = MLPRegression(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001) # lr=0.005

    num_batches = len(X_train_fold) // batch_size

    for epoch in range(num_epochs):
        # Shuffle the training data for each epoch
        indices = torch.randperm(len(X_train_fold))
        X_train_fold = X_train_fold[indices]
        y_train_fold = y_train_fold[indices]

        # Mini-batch training
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Forward pass
            y_pred = model(X_train_fold[start_idx:end_idx])

            # Compute the loss
            loss = criterion(y_pred.squeeze(), y_train_fold[start_idx:end_idx])

            # Backward pass and optimize with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Fold {fold + 1}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Validation on the current fold
    with torch.no_grad():
        y_pred_val = model(X_val_fold)
        val_loss = criterion(y_pred_val.squeeze(), y_val_fold)
        print(f"Validation Loss (Fold {fold + 1}): {val_loss.item():.4f}")

    # Store the trained model for later use on the testing set
    models.append(model)

# Read the CSV file for testing data
test_data = pd.read_csv("aug24_test.csv", header=None)  # Replace "test_data.csv" with the path to your testing CSV file

# Drop rows with missing data (less than 8 columns)
test_data = test_data.dropna(subset=test_data.columns[1:], how='any')
test_data = test_data[test_data.iloc[:, 6] != 1] # drop rows if process number = 1
# test_data = test_data[test_data.iloc[:, 7] != 100] # drop ogbn-products

# Separate the target variable (y) and the input features (X)
X_test_data = test_data.iloc[:, 1:].values
# Normalize the testing data using the same scaler used for training data
X_test_data = scaler.transform(X_test_data)

# Convert testing data to PyTorch tensor
X_test_tensor = torch.tensor(X_test_data, dtype=torch.float32).to(device)

# Inference on the testing set using the average of k models
with torch.no_grad():
    y_test_pred = torch.zeros(len(X_test_data), dtype=torch.float32).to(device)
    for model in models:
        y_test_pred += model(X_test_tensor).squeeze()
    y_test_pred /= num_folds

# Print the predicted values
print("Predicted Values for Testing Set:")
print(y_test_pred)
outp = y_test_pred.tolist()
with open('output_test.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(outp)
