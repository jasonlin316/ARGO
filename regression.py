import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv

# Read the CSV file
data = pd.read_csv("data.csv")  # Replace "data.csv" with the path to your CSV file
data2 = pd.read_csv("test_data.csv")
# Drop rows with missing data (less than 8 columns)
data = data.dropna(subset=data.columns[1:], how='any')

# Separate the target variable (y) and the input features (X)
y_data = data.iloc[:, 0].values
X_data = data.iloc[:, 1:].values

Y_test2 = data2.iloc[:, 0].values
X_test2 = data2.iloc[:, 1:].values
# Normalize the data (subtract mean and divide by standard deviation)
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

X_test2 = scaler.fit_transform(X_test2)

# Split the normalized data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
X_test2_tensor = torch.tensor(X_test2, dtype=torch.float32)
Y_test2_tensor = torch.tensor(Y_test2, dtype=torch.float32)

# Define the multi-layer perceptron model with ReLU activation for non-negativity
class MLPRegression(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegression, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)  # 6 input features and 64 hidden units
        self.layer2 = nn.Linear(128, 1)         # 64 hidden units and 1 output (scalar)

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # ReLU activation for non-negativity
        return torch.relu(self.layer2(x))  # ReLU activation for non-negativity

# Create the model and specify the loss function and optimizer
input_dim = X_train.shape[1]
model = MLPRegression(input_dim)
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.005)

# Gradient Clipping
clip_value = 1.0

# Training the model with mini-batch
batch_size = 16
num_epochs = 500
num_batches = len(X_train_tensor) // batch_size

for epoch in range(num_epochs):
    # Shuffle the training data for each epoch
    indices = torch.randperm(len(X_train_tensor))
    X_train_tensor = X_train_tensor[indices]
    y_train_tensor = y_train_tensor[indices]

    # Mini-batch training
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size

        # Forward pass
        y_pred = model(X_train_tensor[start_idx:end_idx])

        # Compute the loss
        loss = criterion(y_pred.squeeze(), y_train_tensor[start_idx:end_idx])

        # Backward pass and optimize with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)  # Apply gradient clipping
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Testing the model
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    test_loss = criterion(y_pred_test.squeeze(), y_test_tensor)
    print(f"Testing Loss: {test_loss.item():.4f}")

    y_pred_test2 = model(X_test2_tensor)
    
    print(Y_test2_tensor)
    outp = y_pred_test2.tolist()
    # outp = np.transpose(outp)
    # print(outp.item())
    with open('output_test.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(outp)
    
