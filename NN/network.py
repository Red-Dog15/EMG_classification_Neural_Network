"""
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # define layers here
        # input: 3 features for each electrode, output: 6 features
        self.layer1 = nn.Linear(3, 6) # output 6 for now
        self.layer2 = nn.Linear(6, 6) # keep 6 for now
        self.layer3 = nn.Linear(6, 1) # backWW to 3 features for each electrode
        # initialize activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
        """
        
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN_GRU(nn.Module):
    def __init__(self, num_channels=8, num_classes=4):
        super().__init__()

        # ---- CNN feature extractor ----
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)

        # ---- GRU (quantization friendly) ----
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        # ---- Output head ----
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B, seq, channels)
        x = x.transpose(1, 2)   # -> (B, channels, seq)

        # CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.transpose(1, 2)  # back to (B, seq, features)

        # GRU
        out, _ = self.gru(x)
        final = out[:, -1, :]  # last timestep

        return self.fc(final)
    
    
def quantize_model(model):
    q_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8
    )
    return q_model