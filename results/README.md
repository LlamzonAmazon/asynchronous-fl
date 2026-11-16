# Results Directory

This directory contains the results of all experiments.

## Directory layout

```
results/
├── centralized/
│   └── C1/                             ← Centralized baseline
│       ├── experiment_config.txt
│       ├── last_run.log
│       ├── best_model.pth
│       └── training_curves.png
│
├── sync-federated/
│   ├── client_0_dataset.pkl
│   ├── client_1_dataset.pkl
│   ├── client_2_dataset.pkl
│   ├── test_dataset.pkl
│   │
│   ├── sync_IID_4R_3C_1L/              ← IID, 4 Rounds, 3 Clients, 1 Local Epoch
│   │   ├── experiment_config.txt
│   │   ├── last_run.log
│   │   ├── checkpoints/
│   │   ├── global_model.pth
│   │   ├── sync_IID_4R_3C_1L.png       ← training curves (filename = folder name)
│   │   └── network_metrics.json
│   │
│   ├── sync_nonIID_4R_3C_1L/           ← non-IID, 4 Rounds, 3 Clients, 1 Local Epoch
│   │   └── ...
│   │
│   └── sync_IID_4R_2C_1L/              ← IID, 4 Rounds, 2 Clients, 1 Local Epoch
│       └── ...
│
└── async-federated/
    ├── async_IID_4R_3C_1L_K1/          ← IID, K = 1 (equivalent to synchronous FedAvg)
    │   ├── experiment_config.txt
    │   ├── last_run.log
    │   ├── checkpoints/
    │   ├── global_model.pth
    │   ├── async_IID_4R_3C_1L_K1.png   ← training curves (filename = folder name)
    │   ├── network_metrics.json
    │   ├── round_metrics.json
    │   └── run_metadata.json
    │
    ├── async_IID_4R_3C_1L_K2/          ← IID, 4 Rounds, 3 Clients, 1 Local Epoch, K = 2 (Deep Every 2 Rounds)
    │   └── ...
    ├── async_IID_4R_3C_1L_K4/          ← IID, 4 Rounds, 3 Clients, 1 Local Epoch, K = 4
    │   └── ...
    ├── async_nonIID_4R_3C_1L_K2/       ← non-IID, 4 Rounds, 3 Clients, 1 Local Epoch, K = 2
    │   └── ...
    ├── async_nonIID_4R_3C_1L_K4/       ← non-IID, 4 Rounds, 3 Clients, 1 Local Epoch, K = 4
    │   └── ...
    ├── async_IID_4R_2C_1L_K4/          ← IID, 4 Rounds, 2 Clients, 1 Local Epoch, K = 4
    │   └── ...
    └── async_IID_4R_3C_1L_K4_1Mbps/    ← IID, 4 Rounds, 3 Clients, 1 Local Epoch, K = 4, 1 Mbps link
        └── ...
```

## Naming scheme

Run folders are named `<regime>_<IID|nonIID>_<R>R_<C>C_<L>L[_K<k>][_s<seed>]`: `R` is the number
of rounds, `C` the number of clients, `L` the local epochs per round, `K` the deep-layer
synchronization period (asynchronous regime only), and `s<seed>` the random seed. Runs without a
seed suffix use seed 42. Replicate runs with `FL_SEED` set use the suffix and keep their partitions
in `results/sync-federated_s<seed>/`.
