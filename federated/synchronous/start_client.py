"""
Flower Client Launcher

Starts a single Flower client that connects to the server, trains locally, and sends model updates.

Usage:
    python federated/synchronous/start_client.py --client-id 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import flwr as fl
import torch
from pathlib import Path
import pickle

from models.ecg_cnn import ECGCNN
from federated.synchronous.config import fl_config
from federated.synchronous.flower_client import ECGClient
from utils.tee_log import tee_to_file


def load_client_dataset(client_id: int):
    """Load this client's training dataset"""
    client_data_path = Path(fl_config.RESULTS_DIR) / f'client_{client_id}_dataset.pkl'
    
    if not client_data_path.exists():
        raise FileNotFoundError(
            f"Client {client_id} dataset not found at {client_data_path}. "
            "### MUST RUN DATA PREPARATION FIRST ###"
        )
    
    with open(client_data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset


def main():
    """Start a Flower client"""
    log_file = os.environ.get("FL_LOG_FILE")
    if log_file:
        tee_to_file(log_file, mode="a")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start a Flower client')
    parser.add_argument('--client-id', type=int, required=True,
                       help='Client ID (0 to NUM_CLIENTS-1)')
    args = parser.parse_args()
    
    client_id = args.client_id
    
    print(f"\nClient {client_id}: starting.")
    
    # Validate client ID
    if client_id < 0 or client_id >= fl_config.NUM_CLIENTS:
        raise ValueError(
            f"Invalid client ID: {client_id}. "
            f"Must be between 0 and {fl_config.NUM_CLIENTS-1}"
        )
    
    # Load client's dataset
    trainset = load_client_dataset(client_id)
    print(f"Client {client_id}: loaded {len(trainset)} training samples.")
    
    # Create model
    # Each client has its own model instance
    device = torch.device(fl_config.DEVICE)
    model = ECGCNN(
        num_leads=fl_config.NUM_LEADS,
        dropout_rate=fl_config.DROPOUT_RATE
    )
    
    # Create client
    client = ECGClient(
        client_id=client_id,
        model=model,
        trainset=trainset,
        device=device,
        batch_size=fl_config.BATCH_SIZE,
        learning_rate=fl_config.LEARNING_RATE,
        local_epochs=fl_config.LOCAL_EPOCHS
    )
    
    print(f"\nClient {client_id}: connecting to server (localhost:8080).\n")
    
    # Connect to server and start training
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
        grpc_max_message_length=2147483647,  # Max 32-bit int (~2GB)
    )
    
    print(f"\nClient {client_id}: finished. Disconnected from server.\n")


if __name__ == "__main__":
    main()