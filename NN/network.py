"""
EMG Multi-Task Neural Network for Movement and Severity Classification.

This model predicts:
1. Movement class (7 classes): No movement, Wrist Flexion, Extension, etc.
2. Severity level (3 classes): Light, Medium, Hard

Architecture: CNN + GRU with dual output heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class EMG_MultiTask_Model(nn.Module):
    """
    Multi-task model for EMG classification.
    Predicts both movement type and contraction severity.
    """
    
    def __init__(self, num_channels=8, num_movements=7, num_severities=3, 
                 hidden_size=64, dropout=0.3):
        """
        Args:
            num_channels: Number of EMG channels (default 8)
            num_movements: Number of movement classes (default 7)
            num_severities: Number of severity levels (default 3)
            hidden_size: Size of hidden layers
            dropout: Dropout probability for regularization
        """
        super(EMG_MultiTask_Model, self).__init__()
        
        self.num_channels = num_channels
        self.num_movements = num_movements
        self.num_severities = num_severities
        
        # ---- CNN Feature Extractor ----
        # Input: (batch, seq_len, channels)
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2) # first CNN layer, Kernal size 5 looks at 5 consecutive time samples
        self.bn1 = nn.BatchNorm1d(32) # normalize data
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1) # second CNN layer, padding  adds x zeroes to each end of the input
        self.bn2 = nn.BatchNorm1d(64) 
        self.pool = nn.MaxPool1d(kernel_size=2) # take maximum value from every (kernal_size) consecutive samples for computational efficiency
        
        # ---- Temporal Feature Extraction (GRU) ----
        self.gru = nn.GRU( # Gated Recurrent Unit layer for sequential data
            input_size=64, # each time step has 64 features from CNN
            hidden_size=hidden_size, # size of GRU hidden state
            num_layers=2, # number of stacked GRU layers
            batch_first=True,  # expects input as (batch, seq_len, features)
            dropout=dropout if dropout > 0 else 0, # probability of an element to be zeroed
            bidirectional=True # use both forward and backward GRU
        )
        
        """
        GRU Functionality: 
        - Captures temporal dependencies in sequential EMG data
        - Bidirectional GRU processes data in both time directions
        """
        
        # ---- Shared Feature Layer ----
        self.shared_fc = nn.Linear(hidden_size * 2, 128)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout) # randomly sets 30% of neurons to 0 to prevent overfitting in training
        
        # ---- Task-Specific Output Heads ----
        # Movement classification head
        self.movement_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_movements)
        )
        
        # Severity classification head
        self.severity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_severities)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, num_channels)
            
        Returns:
            movement_logits: (batch_size, num_movements)
            severity_logits: (batch_size, num_severities)
        """
        batch_size = x.size(0)
        
        # Transpose for Conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x))) # implement and normalize first conv layer
        x = self.pool(x) #implement pool layer
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Transpose back for GRU: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # GRU layers
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size*2)
        
        # Take last timestep output 
        last_out = gru_out[:, -1, :]  # (batch, hidden_size*2)
        
        # Shared feature extraction
        shared_features = F.relu(self.shared_fc(last_out))
        shared_features = self.dropout(shared_features)
        
        # Task-specific predictions
        movement_logits = self.movement_head(shared_features)
        severity_logits = self.severity_head(shared_features)
        
        return movement_logits, severity_logits

class LightweightEMG_Model(nn.Module):
    """
    Lightweight version for faster training and inference.
    Good for testing and microcontroller deployment.
    """
    
    def __init__(self, num_channels=8, num_movements=7, num_severities=3,
                 conv1_channels=32, conv2_channels=64, shared_dim=128, dropout=0.25):
        super(LightweightEMG_Model, self).__init__()
        
        # Simple CNN
        self.conv1 = nn.Conv1d(num_channels, conv1_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Shared features
        self.fc_shared = nn.Linear(conv2_channels, shared_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output heads
        self.fc_movement = nn.Linear(shared_dim, num_movements)
        self.fc_severity = nn.Linear(shared_dim, num_severities)
        
    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = x.transpose(1, 2)  # -> (batch, channels, seq_len)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (batch, 32)
        
        shared = F.relu(self.fc_shared(x))
        shared = self.dropout(shared)
        
        movement_logits = self.fc_movement(shared)
        severity_logits = self.fc_severity(shared)
        
        return movement_logits, severity_logits

class EMG_Standard_CNN(nn.Module):
    """
    CNN-only multi-task architecture for EMG movement/severity classification.

    Design intent (NN-B):
        - Inspired by the CNN-first approach discussed in:
            Atzori et al., "Deep Learning with Convolutional Neural Networks Applied
            to Electromyography Data", Frontiers in Neurorobotics, 2016.
    - Uses stacked convolution and pooling blocks only (no recurrent layers).
    - Treats EMG as a 2D signal map (channels x time), matching the CNN-first
      design spirit used in EMG image-like representations in literature.
    - Keeps the same dual-head output interface as NN-A for fair comparison.
    """

    def __init__(self, num_channels=8, num_movements=7, num_severities=3, dropout=0.3):
        super(EMG_Standard_CNN, self).__init__()

        self.num_channels = num_channels
        self.num_movements = num_movements
        self.num_severities = num_severities

        # Input shape expected in forward: (batch, seq_len, channels)
        # We reshape to (batch, 1, channels, seq_len) for 2D convolutions.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.movement_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_movements)
        )

        self.severity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_severities)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, num_channels)

        Returns:
            movement_logits: (batch_size, num_movements)
            severity_logits: (batch_size, num_severities)
        """
        # (B, T, C) -> (B, C, T) -> (B, 1, C, T)
        x = x.transpose(1, 2).unsqueeze(1)

        feat = self.features(x)
        shared = self.shared_fc(feat)

        movement_logits = self.movement_head(shared)
        severity_logits = self.severity_head(shared)
        return movement_logits, severity_logits

def create_model(model_type='full', **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: 'full', 'lightweight', or 'standard_cnn'
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        model: PyTorch model
    """
    aliases = {
        'nn_a': 'full',
        'nn_b': 'standard_cnn',
        'nn_c': 'lightweight'
    }
    model_type = aliases.get(model_type.lower(), model_type.lower())

    if model_type == 'lightweight':
        return LightweightEMG_Model(**kwargs)
    if model_type == 'standard_cnn':
        return EMG_Standard_CNN(**kwargs)
    return EMG_MultiTask_Model(**kwargs)


# Legacy commented code for reference
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
def save_output_sim(output_file, data, Clear=True):
    """
    Save prediction results to a suitable data file for myosuite simulation.
    
    Args:
        output_file: Path to output data file
        data: Data to write (string or convertible to string)
    """
    if not os.path.exists(output_file):
        print(f"Output path {output_file} does not exist.")
        return FileNotFoundError
    else:
        file = open(output_file, "w")
        file.write(str(data))
        if Clear:
            file.close()