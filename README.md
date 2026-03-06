# Asynchronous Weight-Updating Federated Learning

Thesis: **A Study of Asynchronous Weight-Updating Federated Learning for IoT Health Devices**

---

## Project abstract

This thesis studies whether asynchronous, layer-wise weight update scheduling in federated learning can reduce communication and synchronization overhead for resource-constrained IoT health devices while maintaining acceptable ECG classification performance. In contrast to standard synchronous FedAvg, where all clients upload full model parameters every round, the proposed approach temporally decouples shallow and deep layer update cadence so that shallow layers are transmitted frequently while deep layers are sent only on selected rounds. Using a shared ECG CNN and the PTB-XL dataset across three training regimes—centralized training, synchronous FL, and asynchronous FL—we hold architecture, data partitions, and total training budget constant while varying only the update schedule. We then measure total bytes transmitted, number of update messages, participation-adjusted communication cost, and server waiting/straggler effects alongside utility metrics (loss and accuracy/AUROC). The experimental matrix (documented in `experiments/EXPERIMENT_MATRIX.md`) spans IID vs non-IID client splits, different client participation and bandwidth regimes, and multiple shallow:deep update ratios to characterize the trade-offs of asynchronous layer-wise scheduling.

## Dataset (PTB-XL)

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (PhysioNet): large public 12-lead ECG dataset.

- **Task**: Binary classification (e.g. NORM vs abnormal).
- **Splits**: Folds 1–9 train/val, fold 10 test (standard).
- **Signals**: 10 s, 500 Hz, 12 leads → (5000, 12) per recording.
- **Labels**: Diagnostic superclass (NORM, MI, STTC, CD, HYP). Data can be partitioned IID or non-IID across clients for FL.

## Tech stack

- **Python3**
- **PyTorch** – model and training
- **Flower** – federated learning (sync server/clients)
- **NumPy, Pandas** – data loading and preprocessing
- **Matplotlib** – training curves and plots

## File structure

```
asynchronous-fl/
├── centralized/
│   ├── config.py              # Centralized data/model/training config
│   └── train.py               # Centralized ECG CNN training + logging
│
├── federated/
│   ├── synchronous/
│   │   ├── config.py          # FL config (clients, rounds, local epochs, IID/non-IID)
│   │   ├── data_partition.py  # IID and non-IID partitioning across clients
│   │   ├── flower_client.py   # Flower client: local training, parameter exchange
│   │   ├── flower_server.py   # FedAvg strategy, server eval, checkpoints, metrics, plots
│   │   ├── run_fl.py          # Orchestrator: prepare data, start server + clients
│   │   ├── start_server.py    # Launches synchronous Flower server
│   │   └── start_client.py    # Launches one synchronous client (--client-id)
│   │
│   └── asynchronous/
│       ├── README.md          # Async FL method description and usage
│       ├── config.py          # Async FL config; mirrors sync + async schedule knobs
│       ├── schedule.py        # Layer-wise update schedules (e.g., periodic shallow/deep)
│       ├── flower_server.py   # Async FedAvg with shallow/deep split, staleness + comm logs
│       ├── flower_client.py   # Async client; full local train, partial uploads per round type
│       ├── run_fl.py          # Orchestrator: validates sync artifacts, runs async server/clients
│       ├── start_server.py    # Launches async Flower server
│       └── start_client.py    # Launches one async client (--client-id)
│
├── models/
│   └── ecg_cnn.py             # Shared ECG CNN architecture for all regimes
│
├── PTB-XL/                    # PTB-XL dataset (or configure path via DATA_PATH)
│
├── results/
│   ├── README.md              # Description of saved metrics, logs, and plots
│   ├── centralized/           # Centralized training artifacts
│   ├── sync-federated/        # Synchronous FL artifacts (incl. shared partitions)
│   └── async-federated/       # Asynchronous FL artifacts
│
├── experiments/
│   ├── EXPERIMENT_MATRIX.md   # Full experimental matrix (regimes, ratios, bandwidth, IID/non-IID)
│   ├── EXP_A2.md              # Example async experiment spec/report
│   └── REPORT_TEMPLATE.md     # Template for writing experiment reports
│
├── Documents/                 # Thesis documents and progress reports
│
├── utils/
│   ├── tee_log.py             # Tee stdout/stderr to log file
│   └── ...                    # Process monitoring and convenience utilities
│
├── LoadData.py                # PTB-XL loader and fold-based splits
├── requirements.txt
├── .gitignore
└── README.md
```

Results are written to `results/centralized/`, `results/sync-federated/`, and `results/async-federated/` (checkpoints, metrics, plots, logs). Place PTB-XL under `PTB-XL/` at the project root or configure `DATA_PATH` in the configs.

### Running experiments

- **Centralized baseline**  
  `python centralized/train.py`

- **Synchronous FL baseline**  
  `python federated/synchronous/run_fl.py`

- **Asynchronous FL (layer-wise updates)**  
  1. Run synchronous FL once to generate shared partition artifacts in `results/sync-federated/`:  
     `python federated/synchronous/run_fl.py`  
  2. Run the async orchestrator (reuses the same partitions):  
     `python federated/asynchronous/run_fl.py`

---

## Author

**Thomas Llamzon** – Honours Computer Science, Western University
