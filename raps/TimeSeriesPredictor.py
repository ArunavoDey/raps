import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample Training Data (Static Features & Corresponding Time Series)
# X_static: (num_samples, num_features)
# Y_series: (num_samples, time_steps)
"""
num_samples = 1000
time_steps = 10
num_features = 5

X_static = np.random.rand(num_samples, num_features)
Y_series = np.random.rand(num_samples, time_steps)

# Scaling Data
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_static = scaler_X.fit_transform(X_static)
Y_series = scaler_Y.fit_transform(Y_series)

# Convert to Torch Tensors
X_static_tensor = torch.tensor(X_static, dtype=torch.float32)
Y_series_tensor = torch.tensor(Y_series, dtype=torch.float32)

# Define Model
class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimeSeriesPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
"""
import torch
import torch.nn as nn

class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=5):
        super(TimeSeriesPredictor, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply ReLU to all except the last layer
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Output layer without activation
        return x





