"""
Script used to automate the client and server activity in a federated learning network

1. Prepares and partitions data

2. Starts Flower server in background

3. Runs clients one at a time
  - Using the Flower simulation mode took up too much memory (50GB+)
    - Ray's infrastructure overhead 
    - Concurrent execution of multiple clients
    - PyTorch's memory management (caching, gradients)
    - Model + optimizer state duplication
  - Each client would load their entire partitioned dataset into memory simultaneously

4. Saves results

Usage:
    python federated/synchronous/run_fl.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import subprocess
import time
import signal
from pathlib import Path
import pickle
import torch

from LoadData import PTBXLDataLoader
from federated.synchronous.config import fl_config
from federated.synchronous.data_partition import prepare_federated_data
from utils.tee_log import tee_to_file


def prepare_data():
    """Load and partition data, save to disk for server/clients to load"""
    
    print("\n" + "=" * 70)
    print("Data preparation")
    print("=" * 70)
    
    # Create results directory
    Path(fl_config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Check if partitioned datasets already exist
    results_dir = Path(fl_config.RESULTS_DIR)
    test_path = results_dir / 'test_dataset.pkl'
    client_paths = [results_dir / f'client_{i}_dataset.pkl' for i in range(fl_config.NUM_CLIENTS)]
    all_files_exist = test_path.exists() and all(p.exists() for p in client_paths)
    
    if all_files_exist and not fl_config.GENERATE_NEW_PARTITION:
        print("\nUsing existing partitioned datasets.")
        print(f"  {fl_config.NUM_CLIENTS} client datasets and test dataset found.")
        print("  Set GENERATE_NEW_PARTITION=True in config to regenerate.\n")
        return None, None
    
    print("\nLoading PTB-XL data...")
    dataset = PTBXLDataLoader(data_path=fl_config.DATA_PATH)
    X_train, y_train, X_test, y_test = dataset.prepare_data(
        superclass=fl_config.SUPERCLASS,
        test_fold=fl_config.TEST_FOLD
    )
    
    print("Partitioning data...")
    client_datasets, test_dataset = prepare_federated_data(
        X_train, y_train, X_test, y_test,
        num_clients=fl_config.NUM_CLIENTS,
        iid=fl_config.IID
    )
    
    print("Saving datasets to disk...")
    # Save client datasets
    for i, dataset in enumerate(client_datasets):
        client_path = Path(fl_config.RESULTS_DIR) / f'client_{i}_dataset.pkl'
        with open(client_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  Saved client {i} dataset")
    
    # Save test dataset
    test_path = Path(fl_config.RESULTS_DIR) / 'test_dataset.pkl'
    with open(test_path, 'wb') as f:
        pickle.dump(test_dataset, f)
    print(f"  Saved test dataset.\n")
    
    return client_datasets, test_dataset


def start_server():
    """Start the Flower server in a SUBPROCESS"""
    
    print("\n" + "-" * 70)
    print("Starting Flower server")
    print("-" * 70)
    
    # Path to server script
    server_script = Path(__file__).parent / 'start_server.py'
    
    # Start server as subprocess
    # Using subprocess module to run the server script in the background
    # stdout/stderr not captured so output is visible in real-time
    server_process = subprocess.Popen(
        [sys.executable, str(server_script)]
    )
    
    print("Waiting for server to initialize...")
    time.sleep(5)
    
    if server_process.poll() is not None:
        print("\nServer failed to start. Check output above for error details.")
        raise RuntimeError("Server failed to start")
    
    print("Server started.\n")
    return server_process


def start_client_process(client_id: int):
    """Start a client as a background process (non-blocking)"""
    
    client_script = Path(__file__).parent / 'start_client.py'
    
    print(f"Starting client {client_id}...")
    
    # Start client as background process
    process = subprocess.Popen(
        [sys.executable, str(client_script), '--client-id', str(client_id)]
    )
    
    return process


def run_federated_learning():
    """Main FL training loop"""
    
    print("\n" + "=" * 70)
    print("Federated learning")
    print("=" * 70)
    print(f"  Clients:      {fl_config.NUM_CLIENTS}")
    print(f"  Rounds:       {fl_config.NUM_ROUNDS}")
    print(f"  Local epochs: {fl_config.LOCAL_EPOCHS}")
    print(f"  IID:          {fl_config.IID}")
    print()
    
    start_time = time.time()
    
    # Start server
    server_process = start_server()
    client_processes = []
    
    try:
        # Run training rounds
        # NOTE: The server handles rounds internally via Flower
        # Just need to ensure all clients connect for each round
        # The server will print progress for each round
        
        print("\nTraining in progress.\n")
        
        # Start all clients concurrently
        print("Starting clients...")
        client_processes = []
        for client_id in range(fl_config.NUM_CLIENTS):
            process = start_client_process(client_id)
            client_processes.append(process)
            time.sleep(1)  # Stagger starts to reduce memory spike
        
        print(f"All {fl_config.NUM_CLIENTS} clients started. Waiting for training to complete.\n")
        
        # Wait for all clients to finish (they stay connected for all rounds)
        for i, process in enumerate(client_processes):
            process.wait()
            print(f"Client {i} finished.")
        
        print("Waiting for server to complete...")
        server_process.wait()
        
    except KeyboardInterrupt:
        print("\n\nTraining stopped by user.")
        # Terminate all client processes
        print("Cleaning up processes...")
        for process in client_processes:
            process.terminate()
        server_process.terminate()
        server_process.wait()
        print("Cleanup complete.\n")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\nError during training: {e}")
        # Terminate all client processes
        for process in client_processes:
            process.terminate()
        server_process.terminate()
        server_process.wait()
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Federated learning finished.")
    print("=" * 70)
    print(f"  Total time: {total_time:.2f} s")
    print(f"  Model:      {fl_config.MODEL_SAVE_PATH}")
    print("=" * 70)


def main():
    """Main entry point"""
    Path(fl_config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    log_path = Path(fl_config.RESULTS_DIR) / "last_run.log"
    tee_to_file(log_path, mode="w")
    os.environ["FL_LOG_FILE"] = str(log_path)

    print("\n" + "=" * 60)
    print("Synchronous federated learning")
    print("=" * 60)
    print(f"Log file: {log_path}\n")

    # Step 1: Prepare data
    prepare_data()
    
    # Step 2: Run federated learning
    run_federated_learning()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)