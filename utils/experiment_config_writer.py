"""
Write a human-readable experiment_config.txt into a results run folder.

Used by centralized, sync, and async pipelines to record run ID and
key config (IID/non-IID, K, clients per round, local epochs, bandwidth, etc.)
for later reference.
"""

from pathlib import Path
from datetime import datetime


def write_centralized_config(results_dir: Path, config) -> None:
    """Write experiment_config.txt for a centralized run."""
    lines = [
        "# Centralized training — experiment config reminder",
        f"# Generated: {datetime.now().isoformat()}",
        "",
        f"RUN_ID = {getattr(config, 'RUN_ID', 'N/A')}",
        "",
        "DATA",
        f"  DATA_PATH = {config.DATA_PATH}",
        f"  SUPERCLASS = {config.SUPERCLASS}",
        f"  TEST_FOLD = {config.TEST_FOLD}",
        "",
        "TRAINING",
        f"  NUM_EPOCHS = {config.NUM_EPOCHS}",
        f"  BATCH_SIZE = {config.BATCH_SIZE}",
        f"  LEARNING_RATE = {config.LEARNING_RATE}",
        f"  PATIENCE = {config.PATIENCE}",
        f"  RANDOM_SEED = {getattr(config, 'RANDOM_SEED', 'N/A')}",
        "",
        "MODEL",
        f"  NUM_LEADS = {config.NUM_LEADS}",
        f"  DROPOUT_RATE = {config.DROPOUT_RATE}",
    ]
    _write_config_file(results_dir, lines)


def write_sync_config(results_dir: Path, config) -> None:
    """Write experiment_config.txt for a synchronous FL run."""
    lines = [
        "# Synchronous FL (FedAvg) — experiment config reminder",
        f"# Generated: {datetime.now().isoformat()}",
        "",
        f"RUN_ID = {getattr(config, 'RUN_ID', 'N/A')}",
        "",
        "DATA",
        f"  IID = {config.IID}",
        f"  NUM_CLIENTS = {config.NUM_CLIENTS}",
        f"  DATA_PATH = {config.DATA_PATH}",
        f"  SUPERCLASS = {config.SUPERCLASS}",
        f"  TEST_FOLD = {config.TEST_FOLD}",
    ]
    if not config.IID:
        lines.append(f"  DIRICHLET_ALPHA = {getattr(config, 'DIRICHLET_ALPHA', 'N/A')}")
    lines.extend([
        "",
        "FEDERATED",
        f"  NUM_ROUNDS = {config.NUM_ROUNDS}",
        f"  CLIENTS_PER_ROUND = {config.CLIENTS_PER_ROUND}",
        f"  LOCAL_EPOCHS = {config.LOCAL_EPOCHS}",
        "",
        "TRAINING",
        f"  BATCH_SIZE = {config.BATCH_SIZE}",
        f"  LEARNING_RATE = {config.LEARNING_RATE}",
        f"  RANDOM_SEED = {getattr(config, 'RANDOM_SEED', 'N/A')}",
        "",
        "MODEL",
        f"  NUM_LEADS = {config.NUM_LEADS}",
        f"  DROPOUT_RATE = {config.DROPOUT_RATE}",
    ])
    _write_config_file(results_dir, lines)


def write_async_config(results_dir: Path, config) -> None:
    """Write experiment_config.txt for an asynchronous FL run."""
    lines = [
        "# Asynchronous layer-wise FL — experiment config reminder",
        f"# Generated: {datetime.now().isoformat()}",
        "",
        f"RUN_ID = {getattr(config, 'RUN_ID', 'N/A')}",
        "",
        "DATA",
        f"  IID = {config.IID}",
        f"  NUM_CLIENTS = {config.NUM_CLIENTS}",
        f"  SYNC_DATA_DIR = {config.SYNC_DATA_DIR}",
        f"  SUPERCLASS = {config.SUPERCLASS}",
        f"  TEST_FOLD = {config.TEST_FOLD}",
        "",
        "FEDERATED",
        f"  NUM_ROUNDS = {config.NUM_ROUNDS}",
        f"  CLIENTS_PER_ROUND = {config.CLIENTS_PER_ROUND}",
        f"  LOCAL_EPOCHS = {config.LOCAL_EPOCHS}",
        "",
        "ASYNC (primary experimental variables)",
        f"  SCHEDULE_TYPE = {config.SCHEDULE_TYPE}",
        f"  DEEP_EVERY_N_ROUNDS (K) = {config.DEEP_EVERY_N_ROUNDS}",
        f"  SHALLOW_PREFIXES = {config.SHALLOW_PREFIXES}",
        f"  SIMULATED_BANDWIDTH_BPS = {config.SIMULATED_BANDWIDTH_BPS}",
        "",
        "TRAINING",
        f"  BATCH_SIZE = {config.BATCH_SIZE}",
        f"  LEARNING_RATE = {config.LEARNING_RATE}",
        f"  RANDOM_SEED = {getattr(config, 'RANDOM_SEED', 'N/A')}",
        "",
        "MODEL",
        f"  NUM_LEADS = {config.NUM_LEADS}",
        f"  DROPOUT_RATE = {config.DROPOUT_RATE}",
    ]
    _write_config_file(results_dir, lines)


def _write_config_file(results_dir: Path, lines: list) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "experiment_config.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Experiment config reminder saved: {path}")
