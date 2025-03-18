import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

class SimpleFeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleFeedForwardNN, self).__init__()

        # Feedforward Network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        scores = self.network(x)
        return scores  # No negation to ensure proper ordering

# Training function
def train_model(model, data, targets, epochs=500, lr=0.01):
    criterion = nn.MSELoss()  # Using MSE loss to fit direct regression targets
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(data).squeeze()
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

