"""
ECG CNN Model for Binary Classification (Normal vs Abnormal)

This model processes 12-lead ECG signals and predicts whether the heart is normal or abnormal.
Architecture is designed with clear shallow/deep layer separation for async FL strategy.

Input: (batch_size, 5000, 12) - ECG signals
Output: (batch_size, 1) - Probability of being normal [0.0 to 1.0]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGCNN(nn.Module):
    """
    1D Convolutional Neural Network for ECG Classification
    
    Architecture:
    - Shallow layers (Conv1, Conv2): Detect basic waveform features
    - Deep layers (Conv3, Conv4, Dense): Combine features into heartbeat patterns,
      learn diagnosis rules and make predictions. The layer split used by the
      asynchronous schedule (SHALLOW_PREFIXES) treats Conv3 as deep.
    """
    
    def __init__(self, num_leads=12, dropout_rate=0.5):
        """
        Initialize the ECG CNN model
        
        Args:
            num_leads: Number of ECG leads (default: 12)
            dropout_rate: Dropout probability to prevent overfitting (default: 0.5)
        """
        super(ECGCNN, self).__init__()
        
        # ============================================================
        # SHALLOW LAYERS - Detects basic patterns (spikes, dips, curves)
        # Uploaded every round in asynchronous FL
        # ============================================================
        
        # Conv1: Looks at very short time windows (0.1 seconds = 50 samples)
        # Input: (batch, 5000, 12)
        # Output: (batch, 2501, 32)
        self.conv1 = nn.Conv1d(
            in_channels=num_leads, # 12 input leads
            out_channels=32, # Learn 32 different basic patterns
            kernel_size=50, # Look at 50 time points at once (0.1 sec)
            stride=2, # Move window by 2 (downsampling)
            padding=25 # Padding edges so size works out
        )
        self.bn1 = nn.BatchNorm1d(32) # Normalize for stable training
        
        # Conv2: Looks at slightly longer windows (0.2 seconds = 100 samples)
        # Output: (batch, 1251, 64)
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64, # Learn 64 more complex patterns
            kernel_size=100, # 0.2 second windows
            stride=2,
            padding=50
        )
        self.bn2 = nn.BatchNorm1d(64)
        
        # ============================================================
        # CONV3 - Combines patterns into heartbeat structures
        # Part of the deep group in the layer split
        # ============================================================
        
        # Conv3: Looks at 1-second windows (500 samples)
        # Output: (batch, 626, 128)
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128, # Learn 128 heartbeat patterns
            kernel_size=500, # 1 second windows (full heartbeat)
            stride=2,
            padding=250
        )
        self.bn3 = nn.BatchNorm1d(128)
        
        # ============================================================
        # DEEP LAYER - Learns rhythm analysis and diagnosis rules
        # Uploaded only on full rounds in asynchronous FL
        # ============================================================
        
        # Conv4: Looks at 2-second windows (1000 samples)
        # Output: (batch, 314, 256)
        self.conv4 = nn.Conv1d(
            in_channels=128,
            out_channels=256, # Learn 256 complex rhythm patterns
            kernel_size=1000, # 2 second windows (multiple beats)
            stride=2,
            padding=500
        )
        self.bn4 = nn.BatchNorm1d(256)
        
        # Global Average Pooling: Looks at ENTIRE 10-second recording
        # Collapses time dimension: (batch, 314, 256) -> (batch, 256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense (Fully Connected) Layers: Make final diagnosis decision
        self.fc1 = nn.Linear(256, 128) # Combine all patterns
        self.dropout1 = nn.Dropout(dropout_rate) # Prevent overfitting
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, 1) # Output is a single probability
        
    def forward(self, x):
        """
        Forward pass: process ECG signal through the network
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, num_leads)
               e.g., (32, 5000, 12)
        
        Returns:
            Output tensor of shape (batch_size, 1) - probability of being normal
        """
        
        # PyTorch Conv1d expects (batch, channels, time)
        # Our input is (batch, time, channels), so we need to transpose
        x = x.transpose(1, 2)  # (batch, 12, 5000)
        
        # ============================================================
        # SHALLOW LAYERS - Extracts basic features
        # ============================================================
        
        # Conv1 + BatchNorm + ReLU activation
        x = self.conv1(x) # (batch, 32, 2501)
        x = self.bn1(x)
        x = F.relu(x) # ReLU: only keep positive activations
        
        # Conv2 + BatchNorm + ReLU
        x = self.conv2(x) # (batch, 64, 1251)
        x = self.bn2(x)
        x = F.relu(x)
        
        # ============================================================
        # CONV3 - Combines into heartbeat patterns
        # ============================================================
        
        x = self.conv3(x) # (batch, 128, 626)
        x = self.bn3(x)
        x = F.relu(x)
        
        # ============================================================
        # DEEP LAYER - Analyze rhythm and diagnose
        # ============================================================
        
        x = self.conv4(x) # (batch, 256, 314)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Global pooling: collapse time dimension
        x = self.global_pool(x) # (batch, 256, 1)
        x = x.squeeze(-1) # (batch, 256) - remove last dimension
        
        # Dense layers for final classification
        x = self.fc1(x) # (batch, 128)
        x = F.relu(x)
        x = self.dropout1(x) # Randomly drop neurons to prevent overfitting
        
        x = self.fc2(x) # (batch, 64)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x) # (batch, 1)
        
        # Sigmoid: convert to probability [0, 1]
        x = torch.sigmoid(x)
        
        return x
    
    def get_shallow_params(self):
        """
        Get parameters from shallow layers (for async FL)
        Returns: iterator of parameters from conv1 and conv2
        """
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.bn1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.bn2.parameters())
        return params
    
    def get_deep_params(self):
        """
        Get parameters from deep layers (for async FL)
        Returns: iterator of parameters from conv3, conv4, and dense layers
        """
        params = []
        params.extend(self.conv3.parameters())
        params.extend(self.bn3.parameters())
        params.extend(self.conv4.parameters())
        params.extend(self.bn4.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params


def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    Useful for understanding model size
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("ECG CNN model test")
    print("=" * 60)

    # Create model
    model = ECGCNN(num_leads=12)
    print(f"\nParameters: {count_parameters(model):,}")
    
    # Test with dummy input
    batch_size = 4
    time_steps = 5000
    num_leads = 12
    
    dummy_input = torch.randn(batch_size, time_steps, num_leads)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output (probabilities): {output.squeeze().detach().numpy()}")
    
    print("\nTest complete.")
    print("\nModel architecture:")
    print(model)