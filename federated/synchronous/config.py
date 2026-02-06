"""
Configuration for Synchronous Federated Learning

All settings for the baseline FL implementation.
"""

import torch

class FLConfig:
    """Federated Learning configuration"""
    
    # DATA SETTINGS
    DATA_PATH = './PTB-XL'
    SUPERCLASS = 'NORM'
    TEST_FOLD = 10
    
    # FEDERATED LEARNING SETTINGS
    # Aligned with centralized: NUM_ROUNDS = effective passes over full (partitioned) data
    # Set NUM_ROUNDS = centralized NUM_EPOCHS for fair comparison (same amount of training).
    NUM_CLIENTS = 5           # Number of simulated IoT devices (reduced for testing)
    NUM_ROUNDS = 15           # Match centralized NUM_EPOCHS (15 effective passes over data)
    CLIENTS_PER_ROUND = 5     # All clients participate each training round
    LOCAL_EPOCHS = 1          # Each client trains once locally per training round

    # Data partitioning
    IID = True                # True: IID split, False: non-IID split
                              # Start with IID for baseline
    GENERATE_NEW_PARTITION = False  # Set to True to regenerate partitioned datasets
                                    # Set to False to reuse existing .pkl files (faster for debugging)
    
    # MODEL SETTINGS
    NUM_LEADS = 12
    DROPOUT_RATE = 0.4
    
    # TRAINING SETTINGS
    BATCH_SIZE = 32           # Batch size for local training
    LEARNING_RATE = 0.001     # Local optimizer learning rate
    
    # DEVICE SETTINGS
    # NOTE: Using CPU for FL because multiple concurrent processes cause MPS GPU contention
    # MPS (Apple Silicon GPU) doesn't handle multiple processes well - causes command buffer errors
    DEVICE = 'cpu'
    # Uncomment below to use MPS for single-process training:
    # if torch.backends.mps.is_available():
    #     DEVICE = 'mps'
    # else:
    #     DEVICE = 'cpu'
    
    # OUTPUT SETTINGS
    RESULTS_DIR = './results/sync-federated'
    MODEL_SAVE_PATH = './results/sync-federated/global_model.pth'
    PLOT_SAVE_PATH = './results/sync-federated/fl_curves.png'
    
    # LOGGING
    VERBOSE = True


# For easy importing
fl_config = FLConfig()