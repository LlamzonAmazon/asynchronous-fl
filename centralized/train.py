"""
Centralized ECG Model Training Script

This script trains the ECG CNN model on the PTB-XL dataset.

This model is used as a baseline for comparison with asynchronous federated learning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from LoadData import PTBXLDataLoader
from models.ecg_cnn import ECGCNN, count_parameters
from centralized.config import config
from utils.tee_log import tee_to_file
from utils.seed import set_seed


def normalize_data(X):
    """
    Normalize ECG signals to zero mean and unit variance to help model train better
    
    Args:
        X: numpy array of shape (n_samples, time_steps, num_leads)
    
    Returns:
        Normalized X with same shape
    """
    # Compute mean and std across all samples and time
    mean = X.mean()
    std = X.std()
    return (X - mean) / (std + 1e-8)  # Avoiding division by zero


def prepare_dataloaders(X_train, y_train, X_test, y_test):
    """
    Convert numpy arrays to PyTorch DataLoaders
    
    DataLoaders handle:
    - Batching (grouping samples)
    - Shuffling (randomizing order)
    - Loading data efficiently
    
    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\nPreparing data loaders...")
    
    # Normalize if configured
    if config.NORMALIZE:
        print("Normalizing data...")
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # Shape: (n, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create training dataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    # Splits training dataset (folds 1-9) into train + validation
    val_size = int(config.VALIDATION_SPLIT * len(train_dataset)) # VALIDATION size (10% of folds 1-9)
    train_size = len(train_dataset) - val_size # TRAINING size (90% of folds 1-9)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size]) # Randomly splits dataset into train and validation
    
    # print(f"Training samples: {train_size}")
    # print(f"Validation samples: {val_size}")
    # print(f"Test samples: {len(X_test)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, # Shuffle training data each epoch
        num_workers=0 # 0 FOR MAC COMPATIBILITY
    )
    
    # Validation dataset loader used to monitor model performance during training
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # Don't shuffle validation
        num_workers=0
    )
    
    # Test dataset loader used to evaluate final model performance
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # Don't shuffle test
        num_workers=0
    )
    
    # Returning train, validation, and test loaders
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch (one pass through all training data)
    
    Args:
        model: The neural network
        train_loader: DataLoader with training data
        criterion: Loss function (measures how wrong predictions are)
        optimizer: Updates model weights
        device: 'mps' or 'cpu'
    
    Returns:
        Average loss and accuracy for this epoch
    """
    model.train()  # Set model to training mode (enables dropout, etc.)
    
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad() # Zero gradients from previous iteration
        
        output = model(data) # FORWARD pass: make predictions
        
        loss = criterion(output, target) # Calculate loss (how wrong the model was)
        
        loss.backward() # BACKWARD pass: compute gradients

        optimizer.step() # Update weights
        
        # Calculating statistics
        total_loss += loss.item()
        predictions = (output > 0.5).float()  # Convert probabilities to 0 or 1
        correct += (predictions == target).sum().item()
        total += target.size(0)
        
        # Prints loss for every 10 batches in the epoch
        # if config.VERBOSE and ((batch_idx + 1) % config.PRINT_EVERY == 0):
        #     print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
        #           f"Loss: {loss.item():.4f}")

        # 5% increment progress bar
        if config.VERBOSE:
            if (batch_idx+1) % (len(train_loader)//20) == 0 or (batch_idx+1) == len(train_loader):
                progress = int(20 * (batch_idx+1) / len(train_loader))
                bar = "=" * progress + " " * (20 - progress)
                percent = 100 * (batch_idx+1) / len(train_loader)

                print(f"\r  Progress: |{bar}| {percent:.0f}% [{batch_idx+1}/{len(train_loader)}]", end="", flush=True)

    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model (no training, just measure performance)
    
    Args:
        model: The neural network
        data_loader: DataLoader with validation or test data
        criterion: Loss function
        device: 'mps' or 'cpu'
    
    Returns:
        Average loss and accuracy
    """
    model.eval()  # Set model to evaluation mode (disables dropout)
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Don't compute gradients (for saving memory and computation)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data) # FORWARD pass
            
            loss = criterion(output, target) # Calculate loss
            
            # Calculating statistics
            total_loss += loss.item()
            predictions = (output > 0.5).float()
            correct += (predictions == target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def visualize(train_losses, val_losses, train_accs, val_accs, test_loss=None, test_acc=None):
    """
    Create plots showing training progress.
    Optionally show final test loss/accuracy so the plot matches printed final results.
    
    Args:
        train_losses, val_losses: Loss values per epoch
        train_accs, val_accs: Accuracy values per epoch
        test_loss, test_acc: Final test set metrics (shown on plot if provided)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    if test_loss is not None:
        ax1.axhline(y=test_loss, color='gray', linestyle='--', alpha=0.8, label=f'Final Test Loss: {test_loss:.4f}')
        ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    if test_acc is not None:
        ax2.axhline(y=test_acc, color='gray', linestyle='--', alpha=0.8, label=f'Final Test Acc: {test_acc:.2f}%')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(config.PLOT_SAVE_PATH)
    print(f"\nTraining curves saved to: {config.PLOT_SAVE_PATH}")


def main():
    """
    Main training function
    """
    # Set global seed for reproducibility
    set_seed(getattr(config, "RANDOM_SEED", None))

    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    tee_to_file(Path(config.RESULTS_DIR) / "last_run.log", mode="w")

    print("=" * 60)
    print("CENTRALIZED ECG MODEL TRAINING")
    print("=" * 60)
    print(f"Log file: {Path(config.RESULTS_DIR) / 'last_run.log'}\n")
    
    # ===== LOAD DATA ============================================
    print("\n### LOADING DATA ###")

    dataset = PTBXLDataLoader(data_path=config.DATA_PATH)

    X_train, y_train, X_test, y_test = dataset.prepare_data(
        superclass=config.SUPERCLASS,
        test_fold=config.TEST_FOLD
    )
    
    # ===== PREPARE DATALOADERS ====================================
    print("\n### PREPARING DATALOADERS ###")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        X_train, y_train, X_test, y_test
    )
    
    # ===== CREATE MODEL ===========================================
    print("\n### CREATING MODEL ###")

    device = torch.device(config.DEVICE)
    model = ECGCNN(num_leads=config.NUM_LEADS, dropout_rate=config.DROPOUT_RATE)
    model = model.to(device)
    
    print(f"Model created with {count_parameters(model):,} parameters")
    print(f"Training on device: {device}")
    
    # ===== SETUP TRAINING =========================================
    print("\n### SETTING UP TRAINING ###")

    # Loss function: Binary Cross Entropy Loss (for binary classification)
    # loss = -[y * log(p) + (1-y) * log(1-p)]
    criterion = nn.BCELoss()
    
    # Optimizer: Adam (adaptive learning rate)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # ===== TRAIN MODEL ============================================
    print("\n### STARTING TRAINING ###")

    print(f"Training for up to {config.NUM_EPOCHS} epochs")
    print(f"Early stopping patience: {config.PATIENCE} epochs\n")
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        
        print(f"\nEPOCH {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"    \nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"    Time: {epoch_time:.1f}s")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"==> New best model saved. (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"==> No improvement ({patience_counter}/{config.PATIENCE})")
            
            if patience_counter >= config.PATIENCE: # Prevents unnecessary training after a certain number of epochs without improvement
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time / 60:.1f} minutes")
    
    # ===== EVALUATE ON TEST SET ===================================
    print("\n### EVALUATING ON TEST SET ###")
    
    # Load best model
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print("-" * 60)
    print("Test set results")
    print("-" * 60)
    print(f"  Test loss:     {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.2f}%")
    print("-" * 60)
    
    # ===== SAVE PLOTS =============================================
    print("\nSaving training curves.")
    visualize(train_losses, val_losses, train_accs, val_accs, test_loss=test_loss, test_acc=test_acc)
    
    print(f"\n{'=' * 60}")
    print("Training finished.")
    print(f"  Results dir:    {config.RESULTS_DIR}")
    print(f"  Best model:    {config.MODEL_SAVE_PATH}")
    print(f"  Curves plot:   {config.PLOT_SAVE_PATH}")
    print(f"  Log file:      {Path(config.RESULTS_DIR) / 'last_run.log'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()