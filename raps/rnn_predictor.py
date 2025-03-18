import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
#import intel_extension_for_pytorch as ipex


class rnn_predictor(nn.Module):
    def __init__(self, input_dim, input_dim2, hidden_dim, output_dim, num_layers):
        super(rnn_predictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_ts = nn.Linear(hidden_dim, output_dim)
        self.fc_num = nn.Linear(input_dim2, output_dim)
        self.fc_combined = nn.Linear(2*output_dim, output_dim)

    def forward(self, X_num, X_ts):
        h0 = torch.zeros(2, X_ts.size(0), 64).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        c0 = torch.zeros(2, X_ts.size(0), 64).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        out_ts, _ = self.lstm(X_ts, (h0, c0))
        out_ts = out_ts[:, -1, :]  # Use the output of the last time step
        
        out_ts = self.fc_ts(out_ts)
        out_num = self.fc_num(X_num)
        print("inside rnn_predictor forward")
        print(f"{out_ts.shape}")
        print(f"{out_num.shape}")
        combined = torch.cat((out_ts, out_num), dim=1)
        out = self.fc_combined(combined)
        return out



