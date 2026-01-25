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


def prepare_data():
    """Load and partition data, save to disk for server/clients to load"""
    
    print("\n" + "#" * 70)
    print("###" + " " * 20 + "FEDERATED LEARNING DATA PREPARATION" + " " * 20 + "###")
    print("#" * 70 + "\n")
    
    # Create results directory
    Path(fl_config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Check if partitioned datasets already exist
    results_dir = Path(fl_config.RESULTS_DIR)
    test_path = results_dir / 'test_dataset.pkl'
    client_paths = [results_dir / f'client_{i}_dataset.pkl' for i in range(fl_config.NUM_CLIENTS)]
    all_files_exist = test_path.exists() and all(p.exists() for p in client_paths)
    
    if all_files_exist and not fl_config.GENERATE_NEW_PARTITION:
        print("\n" + "=" * 70)
        print(">>> USING EXISTING PARTITIONED DATASETS <<<")
        print(f">>> Found {fl_config.NUM_CLIENTS} client datasets and test dataset")
        print(">>> Set GENERATE_NEW_PARTITION=True in config.py to regenerate")
        print("=" * 70 + "\n")
        return None, None
    
    print("\n" + "=" * 70)
    print(">>> LOADING PTB-XL DATA <<<")
    print("=" * 70)
    dataset = PTBXLDataLoader(data_path=fl_config.DATA_PATH)
    X_train, y_train, X_test, y_test = dataset.prepare_data(
        superclass=fl_config.SUPERCLASS,
        test_fold=fl_config.TEST_FOLD
    )
    
    print("\n" + "=" * 70)
    print(">>> PARTITIONING DATA <<<")
    print("=" * 70)
    client_datasets, test_dataset = prepare_federated_data(
        X_train, y_train, X_test, y_test,
        num_clients=fl_config.NUM_CLIENTS,
        iid=fl_config.IID
    )
    
    print("\n" + "=" * 70)
    print(">>> SAVING DATASETS TO DISK <<<")
    print("=" * 70)
    
    # Save client datasets
    for i, dataset in enumerate(client_datasets):
        client_path = Path(fl_config.RESULTS_DIR) / f'client_{i}_dataset.pkl'
        with open(client_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f">>> Saved client {i} dataset")
    
    # Save test dataset
    test_path = Path(fl_config.RESULTS_DIR) / 'test_dataset.pkl'
    with open(test_path, 'wb') as f:
        pickle.dump(test_dataset, f)
    print(f">>> Saved test dataset")
    print("=" * 70 + "\n")
    
    return client_datasets, test_dataset


def start_server():
    """Start the Flower server in a SUBPROCESS"""
    
    print("\n" + "#" * 70)
    print("###" + " " * 24 + "STARTING FLOWER SERVER" + " " * 24 + "###")
    print("#" * 70 + "\n")
    
    # Path to server script
    server_script = Path(__file__).parent / 'start_server.py'
    
    # Start server as subprocess
    # Using subprocess module to run the server script in the background
    # stdout/stderr not captured so output is visible in real-time
    server_process = subprocess.Popen(
        [sys.executable, str(server_script)]
    )
    
    print("\n>>> Waiting for server to initialize...")
    time.sleep(5)
    
    if server_process.poll() is not None:
        print("\n" + "!" * 70)
        print("!!! SERVER FAILED TO START !!!")
        print("!!! Check output above for error details !!!")
        print("!" * 70 + "\n")
        raise RuntimeError("Server failed to start")
    
    print("\n" + "=" * 70)
    print(">>> SERVER STARTED SUCCESSFULLY <<<")
    print("=" * 70 + "\n")
    return server_process


def start_client_process(client_id: int):
    """Start a client as a background process (non-blocking)"""
    
    client_script = Path(__file__).parent / 'start_client.py'
    
    print(f">>> Starting Client {client_id}...")
    
    # Start client as background process
    process = subprocess.Popen(
        [sys.executable, str(client_script), '--client-id', str(client_id)]
    )
    
    return process


def run_federated_learning():
    """Main FL training loop"""
    
    print("\n" + "#" * 70)
    print("###" + " " * 21 + "STARTING FEDERATED LEARNING" + " " * 21 + "###")
    print("#" * 70)
    print(f">>> Configuration:")
    print(f">>>   Clients: {fl_config.NUM_CLIENTS}")
    print(f">>>   Rounds: {fl_config.NUM_ROUNDS}")
    print(f">>>   Local epochs: {fl_config.LOCAL_EPOCHS}")
    print(f">>>   IID: {fl_config.IID}")
    print("#" * 70 + "\n")
    
    start_time = time.time()
    
    # Start server
    server_process = start_server()
    client_processes = []
    
    try:
        # Run training rounds
        # NOTE: The server handles rounds internally via Flower
        # Just need to ensure all clients connect for each round
        # The server will print progress for each round
        
        print("\n" + "=" * 70)
        print(">>> TRAINING IN PROGRESS <<<")
        print("=" * 70 + "\n")
        
        # Start all clients concurrently
        # They will connect to the server and participate in all rounds
        # Server manages rounds internally via Flower
        print("\n>>> Starting all clients...")
        client_processes = []
        for client_id in range(fl_config.NUM_CLIENTS):
            process = start_client_process(client_id)
            client_processes.append(process)
            time.sleep(1)  # Stagger starts to reduce memory spike
        
        print(f"\n>>> All {fl_config.NUM_CLIENTS} clients started")
        print(">>> Waiting for training to complete...")
        print("=" * 70 + "\n")
        
        # Wait for all clients to finish (they stay connected for all rounds)
        for i, process in enumerate(client_processes):
            process.wait()
            print(f"\n>>> Client {i} finished")
        
        # Wait for server to finish
        print("\n>>> Waiting for server to complete...")
        server_process.wait()
        
    except KeyboardInterrupt:
        print("\n\n" + "!" * 70)
        print("!!! TRAINING STOPPED !!!")
        print("!" * 70 + "\n")
        # Terminate all client processes
        print("\n>>> Cleaning up processes...")
        for process in client_processes:
            process.terminate()
        server_process.terminate()
        server_process.wait()
        print(">>> Cleanup complete\n")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\nERROR DURING TRAINING: {e}")
        # Terminate all client processes
        for process in client_processes:
            process.terminate()
        server_process.terminate()
        server_process.wait()
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    print("\n" + "#" * 70)
    print("###" + " " * 20 + "FEDERATED LEARNING COMPLETE" + " " * 20 + "###")
    print("#" * 70)
    print(f"\n>>> Total training time: {total_time:.2f} seconds")
    print(f">>> Model saved to: {fl_config.MODEL_SAVE_PATH}")
    print("=" * 60)


def main():
    """Main entry point"""
    
    print("\n" + "=" * 60)
    print("SYNCHRONOUS FEDERATED LEARNING")
    print("=" * 60)
    
    # Step 1: Prepare data
    prepare_data()
    
    # Step 2: Run federated learning
    run_federated_learning()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTRAINING STOPPED")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)