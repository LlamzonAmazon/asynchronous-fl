"""
Configuration for Asynchronous Layer-Wise Federated Learning

Mirrors federated/synchronous/config.py for all model/training/data settings.
Adds async-specific knobs: schedule type, deep round cadence, participation,
bandwidth simulation, and shallow/deep layer prefix mapping.
"""

import torch


class AsyncFLConfig:
    """Asynchronous Federated Learning configuration"""

    # ── DATA SETTINGS (identical to sync) ──────────────────────────────
    DATA_PATH = './PTB-XL'
    SUPERCLASS = 'NORM'
    TEST_FOLD = 10

    # ── FEDERATED LEARNING SETTINGS ────────────────────────────────────
    # Training budget is identical to sync: NUM_ROUNDS * LOCAL_EPOCHS
    NUM_CLIENTS = 5
    NUM_ROUNDS = 15            # Same as sync — training budget parity
    CLIENTS_PER_ROUND = 5      # Default: all clients participate each round
    LOCAL_EPOCHS = 1           # Each client trains once locally per round

    # Data partitioning
    IID = True
    # Async experiments reuse the same IID / non-IID partitions generated
    # under the synchronous FL config. DIRICHLET_ALPHA is recorded here
    # for completeness and should match federated/synchronous/config.py.
    DIRICHLET_ALPHA = 0.5

    # ── MODEL SETTINGS (identical to sync) ─────────────────────────────
    NUM_LEADS = 12
    DROPOUT_RATE = 0.4

    # ── TRAINING SETTINGS (identical to sync) ──────────────────────────
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EARLY_STOPPING = False    # Global (server-side) early stopping toggle
    PATIENCE = 5              # Patience in rounds for global early stopping
    RANDOM_SEED = 42          # Global random seed for async FL experiments

    # ── DEVICE SETTINGS (identical to sync) ────────────────────────────
    DEVICE = 'cpu'

    # ── ASYNC-SPECIFIC: SCHEDULE SETTINGS ──────────────────────────────
    # Primary experimental variable — controls which rounds are "full" vs
    # "shallow_only". See schedule.py for available schedule types.
    # In the notation used in the thesis, K = DEEP_EVERY_N_ROUNDS.
    SCHEDULE_TYPE = 'periodic'          # 'periodic' | 'warmup_then_periodic' | 'adaptive_plateau'
    DEEP_EVERY_N_ROUNDS = 4             # Deep-layer sync period K for periodic / warmup_then_periodic
    WARMUP_ROUNDS = 0                   # For warmup_then_periodic schedule
    ADAPTIVE_PATIENCE = 3               # For adaptive_plateau schedule
    ADAPTIVE_MIN_GAP = 2                # Minimum gap between deep rounds (adaptive)
    ADAPTIVE_MAX_GAP = 8                # Maximum gap between deep rounds (adaptive)

    # ── ASYNC-SPECIFIC: LAYER SPLIT ────────────────────────────────────
    # Prefixes that identify "shallow" layers in model.state_dict() keys.
    # Everything else is "deep". Never hardcode indices — compute at runtime.
    SHALLOW_PREFIXES = ('conv1.', 'bn1.', 'conv2.', 'bn2.')

    # ── PARTICIPATION & BANDWIDTH KNOBS ────────────────────────────────
    PARTICIPATION_SEED = 42             # Deterministic client sampling seed
    SIMULATED_BANDWIDTH_BPS = None      # None = unlimited; set for bandwidth regime

    # ── DATA SOURCE ────────────────────────────────────────────────────
    # Reuse sync's partitioned .pkl files so both baselines see identical data
    SYNC_DATA_DIR = './results/sync-federated'

    # ── OUTPUT SETTINGS ────────────────────────────────────────────────
    RESULTS_DIR = './results/async-federated'
    MODEL_SAVE_PATH = './results/async-federated/global_model.pth'
    PLOT_SAVE_PATH = './results/async-federated/fl_curves.png'

    # ── LOGGING ────────────────────────────────────────────────────────
    VERBOSE = True


# For easy importing
async_fl_config = AsyncFLConfig()
