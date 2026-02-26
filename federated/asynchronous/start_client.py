"""
Flower Client Launcher — Asynchronous Layer-Wise FL

Mirrors federated/synchronous/start_client.py.

  - Instantiates AsyncECGClient (no shallow_idxs in constructor —
    received from server each round)
  - Loads partition data from SYNC_DATA_DIR (shared artifacts)

Usage:
    python federated/asynchronous/start_client.py --client-id 0
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
from federated.asynchronous.config import async_fl_config
from federated.asynchronous.flower_client import AsyncECGClient
from utils.tee_log import tee_to_file
from utils.seed import set_seed


def load_client_dataset(client_id: int):
    """Load this client's training dataset from sync data directory."""
    client_data_path = (
        Path(async_fl_config.SYNC_DATA_DIR) / f"client_{client_id}_dataset.pkl"
    )

    if not client_data_path.exists():
        raise FileNotFoundError(
            f"Client {client_id} dataset not found at {client_data_path}. "
            "Run synchronous FL first (python federated/synchronous/run_fl.py) "
            "to generate partition artifacts."
        )

    with open(client_data_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def main():
    """Start a Flower client for async FL."""
    log_file = os.environ.get("FL_LOG_FILE")
    if log_file:
        tee_to_file(log_file, mode="a")

    parser = argparse.ArgumentParser(description="Start an async Flower client")
    parser.add_argument(
        "--client-id", type=int, required=True,
        help="Client ID (0 to NUM_CLIENTS-1)",
    )
    args = parser.parse_args()
    client_id = args.client_id

    config = async_fl_config

    print(f"\nAsync Client {client_id}: starting.")

    # Set client-side seed (same base seed for now for reproducibility)
    set_seed(getattr(config, "RANDOM_SEED", None))

    if client_id < 0 or client_id >= config.NUM_CLIENTS:
        raise ValueError(
            f"Invalid client ID: {client_id}. "
            f"Must be between 0 and {config.NUM_CLIENTS - 1}"
        )

    # Load client's dataset (from sync partition artifacts)
    trainset = load_client_dataset(client_id)
    print(f"Client {client_id}: loaded {len(trainset)} training samples.")

    # Create model (same architecture as server)
    device = torch.device(config.DEVICE)
    model = ECGCNN(
        num_leads=config.NUM_LEADS,
        dropout_rate=config.DROPOUT_RATE,
    )

    # Create async client
    client = AsyncECGClient(
        client_id=client_id,
        model=model,
        trainset=trainset,
        device=device,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        local_epochs=config.LOCAL_EPOCHS,
    )

    print(f"\nClient {client_id}: connecting to server (localhost:8080).\n")

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
        grpc_max_message_length=2147483647,
    )

    print(f"\nClient {client_id}: finished. Disconnected from server.\n")


if __name__ == "__main__":
    main()
