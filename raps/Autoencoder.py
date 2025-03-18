import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
#import wandb
import torch
class ImprovedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, attention_dim=32):
        super(ImprovedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, attention_dim),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=1)
        self.decoder = nn.Sequential(
            nn.Linear(attention_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(0)  # Add a batch dimension for the attention mechanism
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)  # Remove the batch dimension
        x = self.decoder(x)
        return x

    def train_improved_autoencoder_plain_data(self, data, input_dim, encoding_dim, attention_dim=32, epochs=50, batch_size=256, validation_split=0.2):      
        train_size = int((1 - validation_split) * len(data))
        val_size = len(data) - train_size
        train_data, val_data = random_split(TensorDataset(torch.tensor(data, dtype=torch.float32)), [train_size, val_size])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        #input_dim = data.shape[1]
        print('INPUT DIM OF FEATURES: ', input_dim, data.shape[1])
        model = ImprovedAutoencoder(input_dim, encoding_dim, attention_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                inputs = batch[0]
                inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                optimizer.zero_grad()
                reconstruction = model(inputs)
                loss = criterion(reconstruction, inputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0]
                    inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    reconstruction = model(inputs)
                    loss = criterion(reconstruction, inputs)
                    val_loss += loss.item()
        
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            #wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        #wandb.finish()
        encoder = model.encoder
        return encoder, train_losses, val_losses, model, train_loader, val_loader


