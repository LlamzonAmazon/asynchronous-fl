'''
Config file for centralized ECG model training
'''

class Config:
    """Training configuration"""
    
    # DATA SETTINGS
    DATA_PATH = './PTB-XL' 
    SUPERCLASS = 'NORM' # NORM vs abnormal
    TEST_FOLD = 10 # Use fold 10 as test set (standard)
    
    # MODEL SETTINGS
    NUM_LEADS = 12 # ECG leads
    DROPOUT_RATE = 0.4 # Dropout probability
    
    # TRAINING SETTINGS
    BATCH_SIZE = 32 # Number of samples per batch
    
    NUM_EPOCHS = 15 # Maximum number of training epochs (match FL NUM_ROUNDS for fair comparison)
    
    LEARNING_RATE = 0.001 # How fast the model learns
    
    PATIENCE = 3 # Early stopping: stop if no improvement for 3 epochs
    RANDOM_SEED = 42 # Global random seed for reproducible centralized experiments
    
    # DEVICE SETTINGS
    # Automatically use MPS (Apple Silicon GPU) if available, otherwise CPU
    import torch
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
        print("\nDevice: MPS (Apple Silicon GPU)")
    else:
        DEVICE = 'cpu'
        print("\nDevice: CPU")
    
    # OUTPUT SETTINGS
    RESULTS_DIR = './results/centralized'
    MODEL_SAVE_PATH = './results/centralized/best_model.pth'
    PLOT_SAVE_PATH = './results/centralized/training_curves.png'
    
    # DATA PREPROCESSING
    NORMALIZE = True # Whether to normalize ECG signals
    VALIDATION_SPLIT = 0.1 # 10% of training data (fold 10) used for validation
    
    # LOGGING
    PRINT_EVERY = 10 # Print Epoch progress every N batches
    VERBOSE = True # Print detailed training info

config = Config()