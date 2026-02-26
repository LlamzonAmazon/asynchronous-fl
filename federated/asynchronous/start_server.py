"""
Flower Server Launcher — Asynchronous Layer-Wise FL

Mirrors federated/synchronous/start_server.py.

  - Computes ALL_KEYS, SHALLOW_IDXS, DEEP_IDXS from model state_dict
  - Creates schedule via create_schedule(config)
  - Instantiates AsyncLayerFedAvg strategy
  - After training: saves run_metadata, network_metrics, training curves

Usage:
    python federated/asynchronous/start_server.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import flwr as fl
import torch
from pathlib import Path
import pickle
import shutil

from models.ecg_cnn import ECGCNN
from federated.asynchronous.config import async_fl_config
from federated.asynchronous.schedule import create_schedule
from federated.asynchronous.flower_server import (
    AsyncLayerFedAvg,
    compute_layer_split,
    get_initial_parameters,
    plot_training_curves,
    save_network_metrics,
    save_run_metadata,
    weighted_average,
)
from torch.utils.data import DataLoader
from utils.tee_log import tee_to_file
from utils.seed import set_seed


def load_test_dataset():
    """Load test dataset from sync data directory (shared partition artifacts)."""
    test_data_path = Path(async_fl_config.SYNC_DATA_DIR) / "test_dataset.pkl"

    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found at {test_data_path}. "
            "Run synchronous FL first (python federated/synchronous/run_fl.py) "
            "to generate partition artifacts."
        )

    with open(test_data_path, "rb") as f:
        test_dataset = pickle.load(f)
    return test_dataset


def save_final_model_backup(model, config):
    """Backup: copy last checkpoint to global_model.pth if evaluate_fn didn't."""
    checkpoint_dir = Path(config.RESULTS_DIR) / "checkpoints"
    final_checkpoint = checkpoint_dir / f"model_round_{config.NUM_ROUNDS}.pth"
    final_model_path = Path(config.MODEL_SAVE_PATH)

    if final_checkpoint.exists():
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_checkpoint, final_model_path)
        print(f"Final global model saved: {final_model_path}")
    else:
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_model_path)
        print(f"Final global model saved: {final_model_path}")


def main():
    """Start the async Flower server."""
    log_file = os.environ.get("FL_LOG_FILE")
    if log_file:
        tee_to_file(log_file, mode="a")

    config = async_fl_config

    # Set global seed for reproducible async server behaviour
    set_seed(getattr(config, "RANDOM_SEED", None))

    print("\n" + "=" * 70)
    print("Async Layer-Wise Flower Server")
    print("=" * 70)

    # ── Create global model ──────────────────────────────────────────
    device = torch.device(config.DEVICE)
    global_model = ECGCNN(
        num_leads=config.NUM_LEADS,
        dropout_rate=config.DROPOUT_RATE,
    )
    print(f"Model device: {device}")

    # ── Load test dataset ────────────────────────────────────────────
    print("Loading test dataset...")
    test_dataset = load_test_dataset()
    print(f"Test dataset: {len(test_dataset)} samples")

    # ── Compute layer split (single source of truth) ─────────────────
    split = compute_layer_split(global_model, config)
    all_keys = split["all_keys"]
    shallow_idxs = split["shallow_idxs"]
    deep_idxs = split["deep_idxs"]
    all_keys_hash = split["all_keys_hash"]

    print(f"Layer split: {len(all_keys)} total keys, "
          f"{len(shallow_idxs)} shallow, {len(deep_idxs)} deep")
    print(f"Byte sizes: shallow={split['shallow_bytes']:,} B, "
          f"deep={split['deep_bytes']:,} B, total={split['full_bytes']:,} B")
    print(f"Shallow fraction: {split['shallow_bytes'] / split['full_bytes'] * 100:.1f}%")

    # ── Create schedule ──────────────────────────────────────────────
    schedule = create_schedule(config)
    print(f"Schedule: {schedule.description()}")

    # ── Get initial parameters ───────────────────────────────────────
    initial_parameters = get_initial_parameters(global_model)

    # ── Create test loader for evaluation ────────────────────────────
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    eval_config = {
        "model": global_model,
        "test_loader": test_loader,
        "device": device,
        "results_dir": config.RESULTS_DIR,
        "num_rounds": config.NUM_ROUNDS,
        "model_save_path": config.MODEL_SAVE_PATH,
    }

    # ── Create AsyncLayerFedAvg strategy ─────────────────────────────
    print("Creating AsyncLayerFedAvg strategy...")
    strategy = AsyncLayerFedAvg(
        schedule=schedule,
        all_keys=all_keys,
        shallow_idxs=shallow_idxs,
        deep_idxs=deep_idxs,
        all_keys_hash=all_keys_hash,
        eval_config=eval_config,
        config=config,
        # FedAvg base kwargs
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=config.CLIENTS_PER_ROUND,
        min_evaluate_clients=0,
        min_available_clients=config.NUM_CLIENTS,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
    )

    # ── Save run metadata before training ────────────────────────────
    save_run_metadata(
        results_dir=config.RESULTS_DIR,
        config=config,
        all_keys=all_keys,
        shallow_idxs=shallow_idxs,
        deep_idxs=deep_idxs,
        all_keys_hash=all_keys_hash,
        schedule_desc=schedule.description(),
        full_model_bytes=split["full_bytes"],
        shallow_bytes=split["shallow_bytes"],
        deep_bytes=split["deep_bytes"],
    )

    print("\n" + "-" * 70)
    print("Server ready. Waiting for clients.")
    print(f"  Clients required: {config.NUM_CLIENTS}")
    print(f"  Rounds:           {config.NUM_ROUNDS}")
    print(f"  Schedule:         {config.SCHEDULE_TYPE} (N={config.DEEP_EVERY_N_ROUNDS})")
    print("-" * 70 + "\n")

    # ── Start Flower server ──────────────────────────────────────────
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy,
        grpc_max_message_length=2147483647,
    )

    # ── Post-training outputs ────────────────────────────────────────
    save_final_model_backup(global_model, config)

    plot_training_curves(
        config.RESULTS_DIR,
        config.PLOT_SAVE_PATH,
        round_log=strategy.round_log,
    )

    save_network_metrics(
        round_log=strategy.round_log,
        results_dir=config.RESULTS_DIR,
        config=config,
        full_model_bytes=split["full_bytes"],
        shallow_bytes=split["shallow_bytes"],
        deep_bytes=split["deep_bytes"],
    )

    print("\n" + "=" * 70)
    print("Server: training complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
