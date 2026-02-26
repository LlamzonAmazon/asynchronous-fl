"""
Flower Server Launcher

Starts the Flower server that coordinates federated learning.
The server waits for clients to connect, aggregates their updates,
and evaluates the global model.

Usage:
    python federated/synchronous/start_server.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import flwr as fl
import torch
from pathlib import Path

from models.ecg_cnn import ECGCNN
from federated.synchronous.config import fl_config
from federated.synchronous.flower_server import (
    create_strategy,
    get_initial_parameters,
    plot_training_curves,
    save_network_metrics,
)
from torch.utils.data import TensorDataset, DataLoader
from utils.tee_log import tee_to_file
from utils.seed import set_seed
import pickle
import shutil


def load_test_dataset():
    """Load the test dataset that was saved during data preparation"""
    test_data_path = Path(fl_config.RESULTS_DIR) / 'test_dataset.pkl'
    
    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found at {test_data_path}. "
            "Run data preparation first!"
        )
    
    with open(test_data_path, 'rb') as f:
        test_dataset = pickle.load(f)
    
    return test_dataset


def save_final_model_backup(model, config):
    """
    Backup method to save final model by copying the last checkpoint.
    This ensures global_model.pth exists even if evaluate_fn didn't save it.
    """
    checkpoint_dir = Path(config.RESULTS_DIR) / "checkpoints"
    final_checkpoint = checkpoint_dir / f"model_round_{config.NUM_ROUNDS}.pth"
    final_model_path = Path(config.MODEL_SAVE_PATH)
    
    if final_checkpoint.exists():
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_checkpoint, final_model_path)
        print(f"Final global model saved: {final_model_path}")
    else:
        # If checkpoint doesn't exist, save current model state
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_model_path)
        print(f"Final global model saved: {final_model_path}")


def main():
    """Start the Flower server"""
    log_file = os.environ.get("FL_LOG_FILE")
    if log_file:
        tee_to_file(log_file, mode="a")

    # Set global seed for reproducible server-side behaviour
    set_seed(getattr(fl_config, "RANDOM_SEED", None))

    print("\n" + "=" * 70)
    print("Flower server")
    print("=" * 70)
    
    # Create global model
    device = torch.device(fl_config.DEVICE)
    global_model = ECGCNN(
        num_leads=fl_config.NUM_LEADS,
        dropout_rate=fl_config.DROPOUT_RATE
    )
    
    print(f"Model device: {device}")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_test_dataset()
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Get initial parameters
    initial_parameters = get_initial_parameters(global_model)
    
    # Create strategy (FedAvg)
    print("Creating FedAvg strategy...")
    strategy = create_strategy(
        model=global_model,
        test_dataset=test_dataset,
        device=device,
        config=fl_config,
        initial_parameters=initial_parameters
    )
    
    print("\n" + "-" * 70)
    print("Server ready. Waiting for clients.")
    print(f"  Clients required: {fl_config.NUM_CLIENTS}")
    print(f"  Rounds:          {fl_config.NUM_ROUNDS}")
    print("-" * 70 + "\n")
    
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=fl_config.NUM_ROUNDS),
        strategy=strategy,
        grpc_max_message_length=2147483647,  # Max 32-bit int (~2GB)
    )
    
    # Ensure final model is saved (backup in case evaluate_fn didn't save it)
    save_final_model_backup(global_model, fl_config)

    # Save training curves (test loss & accuracy per round)
    plot_training_curves(fl_config.RESULTS_DIR, fl_config.PLOT_SAVE_PATH)

    # Document network communication cost for this FL run
    save_network_metrics(
        global_model,
        fl_config.RESULTS_DIR,
        num_rounds=fl_config.NUM_ROUNDS,
        num_clients=fl_config.NUM_CLIENTS,
    )
    
    print("\n" + "=" * 70)
    print("Server: training complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()