"""
Orchestrator for Asynchronous Layer-Wise Federated Learning

Mirrors federated/synchronous/run_fl.py.  Key differences:
  - Reads partition data from SYNC_DATA_DIR (reuses sync's .pkl files)
  - Prints async-specific config summary (schedule, cadence, participation)
  - If sync partition artifacts are missing, raises a clear error

Usage:
    python federated/asynchronous/run_fl.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import subprocess
import time
from pathlib import Path

from federated.asynchronous.config import async_fl_config
from utils.tee_log import tee_to_file
from utils.seed import set_seed


def verify_sync_data():
    """Verify that sync partition artifacts exist before starting."""

    print("\n" + "=" * 70)
    print("Data verification")
    print("=" * 70)

    config = async_fl_config
    sync_dir = Path(config.SYNC_DATA_DIR)

    test_path = sync_dir / "test_dataset.pkl"
    client_paths = [sync_dir / f"client_{i}_dataset.pkl" for i in range(config.NUM_CLIENTS)]

    missing = []
    if not test_path.exists():
        missing.append(str(test_path))
    for p in client_paths:
        if not p.exists():
            missing.append(str(p))

    if missing:
        print("\nERROR: Sync partition artifacts not found:")
        for m in missing:
            print(f"  - {m}")
        print(
            f"\nRun synchronous FL first to generate partition data:\n"
            f"  python federated/synchronous/run_fl.py\n"
        )
        raise FileNotFoundError(
            "Sync partition artifacts missing. "
            "Run synchronous FL first to generate shared data partitions."
        )

    print(f"\nUsing partition data from: {sync_dir}")
    print(f"  {config.NUM_CLIENTS} client datasets + test dataset found.")
    print("  (Shared with sync baseline for fair comparison)\n")


def start_server():
    """Start the async Flower server in a subprocess."""

    print("\n" + "-" * 70)
    print("Starting async Flower server")
    print("-" * 70)

    server_script = Path(__file__).parent / "start_server.py"

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
    """Start a client as a background process."""

    client_script = Path(__file__).parent / "start_client.py"

    print(f"Starting client {client_id}...")

    process = subprocess.Popen(
        [sys.executable, str(client_script), "--client-id", str(client_id)]
    )

    return process


def run_federated_learning():
    """Main async FL training loop."""

    config = async_fl_config

    print("\n" + "=" * 70)
    print("Asynchronous layer-wise federated learning")
    print("=" * 70)
    print(f"  Clients:          {config.NUM_CLIENTS}")
    print(f"  Clients/round:    {config.CLIENTS_PER_ROUND}")
    print(f"  Rounds:           {config.NUM_ROUNDS}")
    print(f"  Local epochs:     {config.LOCAL_EPOCHS}")
    print(f"  IID:              {config.IID}")
    print(f"  Schedule:         {config.SCHEDULE_TYPE}")
    print(f"  Deep every N:     {config.DEEP_EVERY_N_ROUNDS}")
    print(f"  Shallow prefixes: {config.SHALLOW_PREFIXES}")
    print(f"  Participation:    seed={config.PARTICIPATION_SEED}")
    print(f"  Bandwidth sim:    {config.SIMULATED_BANDWIDTH_BPS}")
    print()

    start_time = time.time()

    server_process = start_server()
    client_processes = []

    try:
        print("\nTraining in progress.\n")

        # Start all clients concurrently
        print("Starting clients...")
        for client_id in range(config.NUM_CLIENTS):
            process = start_client_process(client_id)
            client_processes.append(process)
            time.sleep(1)  # Stagger starts to reduce memory spike

        print(
            f"All {config.NUM_CLIENTS} clients started. "
            "Waiting for training to complete.\n"
        )

        # Wait for all clients to finish
        for i, process in enumerate(client_processes):
            process.wait()
            print(f"Client {i} finished.")

        print("Waiting for server to complete...")
        server_process.wait()

    except KeyboardInterrupt:
        print("\n\nTraining stopped by user.")
        print("Cleaning up processes...")
        for process in client_processes:
            process.terminate()
        server_process.terminate()
        server_process.wait()
        print("Cleanup complete.\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nError during training: {e}")
        for process in client_processes:
            process.terminate()
        server_process.terminate()
        server_process.wait()
        sys.exit(1)

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("Asynchronous federated learning finished.")
    print("=" * 70)
    print(f"  Total time: {total_time:.2f} s")
    print(f"  Model:      {config.MODEL_SAVE_PATH}")
    print(f"  Results:    {config.RESULTS_DIR}")
    print("=" * 70)


def main():
    """Main entry point."""
    config = async_fl_config

    # Set global seed for async orchestrator logging / any randomness here
    set_seed(getattr(config, "RANDOM_SEED", None))

    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    log_path = Path(config.RESULTS_DIR) / "last_run.log"
    tee_to_file(log_path, mode="w")
    os.environ["FL_LOG_FILE"] = str(log_path)

    print("\n" + "=" * 60)
    print("Asynchronous layer-wise federated learning")
    print("=" * 60)
    print(f"Log file: {log_path}\n")

    # Step 1: Verify sync partition data exists
    verify_sync_data()

    # Step 2: Run async federated learning
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
