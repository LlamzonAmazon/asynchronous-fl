# Asynchronous Weight-Updating Federated Learning

Thesis: **A Study of Asynchronous Weight-Updating Federated Learning for IoT Health Devices**

---

## Problem statement

In standard federated learning (FL), client devices send full model updates synchronously in every training round. This causes high network usage and idle time on resource-limited devices (e.g. healthcare IoT). Sending both shallow and deep layer updates every round adds unnecessary communication and compute, which is a poor fit for devices that need to focus capacity on local inference and anomaly detection.

## Proposed solution

**Asynchronous weight-updating**: temporally decouple shallow and deep layer parameter updates. Shallow updates are sent more often and deep updates less often. The goal is to reduce network overhead and device load while keeping model accuracy under bandwidth and compute constraints.

## Methodology
All architectures use the some ECG CNN model and dataset to ensure fair comparison.
- **Centralized baseline**: Single-machine training on PTB-XL.
- **Synchronous FL baseline**: FedAvg over Flower. All clients participate each round. This experiment uses the same total “passes” over the data as centralized to ensure fair comparison.
- **Asynchronous FL (planned)**: Same model and dataset with decoupled shallow/deep update schedule. Then the accuracy, communication cost, and runtime vs synchronous FL and centralized will be evaluated and compared against the baselines.

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
│   ├── config.py              # Data, model, training (epochs, LR, early stopping)
│   └── train.py               # Centralized ECG CNN training; logs, plots, best model
│
├── federated/
│   ├── synchronous/
│   │   ├── config.py           # FL config (clients, rounds, local epochs, IID/non-IID)
│   │   ├── data_partition.py   # IID and non-IID partitioning across clients
│   │   ├── flower_client.py    # Flower client: local training, parameter exchange
│   │   ├── flower_server.py   # FedAvg strategy, server eval, checkpoints, metrics, plots
│   │   ├── run_fl.py           # Orchestrator: prepare data, start server, run clients
│   │   ├── start_server.py    # Launches Flower server
│   │   └── start_client.py    # Launches one client (--client-id)
│   │
│   └── asynchronous/
│       ├── #TODO
│       └── 
│
├── models/
│   └── ecg_cnn.py             # ECG CNN (12-lead); used by centralized and FL
│
├── utils/
│   ├── tee_log.py             # Tee stdout/stderr to log file
│   ├── __init__.py
│   └── ...                    # check_processes, kill, monitor (dev/debug)
│
├── LoadData.py                # PTB-XL loader: metadata, signals, train/test split by fold
├── requirements.txt
├── .gitignore
└── README.md
```

Generated at runtime: `results/centralized/`, `results/sync-federated/` (checkpoints, metrics, plots, logs). Dataset: place PTB-XL under project root (or set `DATA_PATH` in configs).

---

## Author

**Thomas Llamzon** – Honours Computer Science, Western University
