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
    NUM_CLIENTS = 3           # Number of simulated IoT devices
    NUM_ROUNDS = 4            # Match centralized NUM_EPOCHS (4 effective passes over data)
    CLIENTS_PER_ROUND = 3     # All clients participate each training round
    LOCAL_EPOCHS = 1          # Each client trains once locally per training round

    # Data partitioning
    IID = True                # True: IID split, False: non-IID split
                              # Start with IID for baseline
    GENERATE_NEW_PARTITION = True   # Regenerate partitioned datasets for new NUM_CLIENTS
                                    # Set to False afterwards to reuse existing .pkl files
    DIRICHLET_ALPHA = 0.5     # Concentration parameter for non-IID Dirichlet splits (see data_partition.py)
    
    # MODEL SETTINGS
    NUM_LEADS = 12
    DROPOUT_RATE = 0.4
    
    # TRAINING SETTINGS
    BATCH_SIZE = 32           # Batch size for local training
    LEARNING_RATE = 0.001     # Local optimizer learning rate
    RANDOM_SEED = 42          # Global random seed for FL experiments
    EARLY_STOPPING = False    # Global (server-side) early stopping toggle
    PATIENCE = 5              # Patience in rounds for global early stopping
    
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
    # Partition .pkl files are stored in PARTITION_DIR (shared; async reads from here).
    # Run outputs go to RESULTS_DIR = PARTITION_DIR / RUN_ID. Set RUN_ID per run (e.g. "sync_IID", "A4_baseline").
    PARTITION_DIR = './results/sync-federated'
    RUN_ID = 'sync_IID'
    RESULTS_DIR = f'{PARTITION_DIR}/{RUN_ID}'
    MODEL_SAVE_PATH = f'{RESULTS_DIR}/global_model.pth'
    PLOT_SAVE_PATH = f'{RESULTS_DIR}/fl_curves.png'
    
    # LOGGING
    VERBOSE = True


# For easy importing
fl_config = FLConfig()