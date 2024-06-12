import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Load the dataset
data = pd.read_csv('D:\Sign-Language-Detection\gestures.csv')

# Sample 1% or 2% of the data for verification
sampled_data = data.sample(frac=0.02, random_state=42)  # Change frac to 0.01 for 1% sampling
print("Sampled Data:")
print(sampled_data.head())

# Extract features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
