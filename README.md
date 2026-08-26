# A Study of Asynchronous Weight-Updating Federated Learning for IoT Health Devices
Thomas Llamzon, Honours Specialization in Computer Science (BSc), Western University

[![Report](https://img.shields.io/badge/Read_My_Thesis-PDF-E74C3C?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](THESIS.pdf)
[![Paper](https://img.shields.io/badge/IEEE_GLOBECOM_2026_Paper-PDF-00629B?style=for-the-badge&logo=ieee&logoColor=white)](paper/main-conference.pdf)

---
## Publication

**Asynchronous Layer-Wise Weight Updating for Communication-Efficient Federated Learning in IoT Health Devices**
T. Llamzon, M. Fouda, M. I. Ibrahem, S. Hashima, Z. Md Fadlullah.
Accepted to **IEEE GLOBECOM 2026** (Selected Areas in Communications: e-Health), Macau, December 2026.

The copy in [`paper/`](paper/) is the **accepted manuscript, camera-ready in review**. It is not the
final published version; the version of record will appear in IEEE Xplore. Source (`main-conference.tex`,
`ref.bib`, `Tikz-Figs/`) is included so the figures can be regenerated from the run artifacts.

© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all
other uses, in any current or future media, including reprinting/republishing this material for
advertising or promotional purposes, creating new collective works, for resale or redistribution to
servers or lists, or reuse of any copyrighted component of this work in other works.

```bibtex
@inproceedings{llamzon2026asyncfl,
  title     = {Asynchronous Layer-Wise Weight Updating for Communication-Efficient Federated Learning in IoT Health Devices},
  author    = {Llamzon, Thomas and Fouda, Mostafa and Ibrahem, Mohamed I. and Hashima, Sherief and Fadlullah, Zubair Md},
  booktitle = {Proc. IEEE Global Communications Conference (GLOBECOM)},
  year      = {2026},
  note      = {Accepted; to appear}
}
```

---
## System Architecture

![Figure 5.1 – System Context Diagram](fl_system_diagram.drawio.png)

---

## The Foundation of Asynchronous Weight-Updating 

- **The CNN model is split into shallow and deep layers**: The global ECG CNN parameters $\boldsymbol{\theta}$ are partitioned into a shallow tensor $\boldsymbol{\theta}_S$ (early layers with prefixes matching `SHALLOW_PREFIXES`) and a deep tensor $\boldsymbol{\theta}_D$ (all remaining layers).

- **Clients always train the full model**: In each communication round $t$, every participating client $i \in \mathcal{P}\_t$ runs local training on its private ECG data and updates both $\boldsymbol{\theta}\_{S,i}^t$ and $\boldsymbol{\theta}\_{D,i}^t$.

- **Server aggregates shallow layers every round**: The server performs a standard FedAvg-style aggregation of the shallow partition every round, so low-level ECG feature extractors stay well aligned across clients.

- **Deep layers synchronize only on selected rounds**: Deep layers are aggregated only in "full" rounds, every $K$-th round ($t \bmod K = 0$). In between, the server reuses a cached copy of the most recently aggregated deep parameters instead of collecting new deep updates.

- **Communication savings**: Uplink messages in shallow-only rounds contain only $|\boldsymbol{\theta}_S|$ bytes instead of $|\boldsymbol{\theta}_S| + |\boldsymbol{\theta}_D|$, reducing client $\rightarrow$ server communication without changing the model architecture or the local training budget.

- **Server-side evaluation only**: After each round, the server evaluates the updated global model on a fixed held-out PTB-XL test set (shared with the centralized and synchronous baselines). No client-side evaluation is performed (`fraction_evaluate=0.0`, `min_evaluate_clients=0`), so all utility curves are directly comparable across regimes.

---

## The Asynchronous Weight-Updating Formulation

We let $T \in \mathbb{N}$ be the total number of communication rounds and $K \in \mathbb{N}$ the deep-layer synchronization period. The model parameters are split as

$$
\boldsymbol{\theta} = (\boldsymbol{\theta}_S,\ \boldsymbol{\theta}_D)
$$

where $\boldsymbol{\theta}\_S$ are the shallow parameters and $\boldsymbol{\theta}\_D$ are the deep parameters. In round $t \in \{1, \dots, T\}$, the set of participating clients is $\mathcal{P}\_t$. After local training, client $i$ holds $\boldsymbol{\theta}\_{S,i}^t$ and $\boldsymbol{\theta}\_{D,i}^t$, and the server computes FedAvg-style aggregates

$$
\text{Agg}_S^t = \sum_{i \in \mathcal{P}_t} w_i^t \boldsymbol{\theta}_{S,i}^t, \qquad
\text{Agg}_D^t = \sum_{i \in \mathcal{P}_t} w_i^t \boldsymbol{\theta}_{D,i}^t, \qquad
\sum_{i \in \mathcal{P}_t} w_i^t = 1
$$

The server maintains a deep-parameter cache $\widehat{\boldsymbol{\theta}}_D^t$. Under the `PeriodicSchedule`, the server update at round $t$ is

$$
\boldsymbol{\theta}_S^{t+1} = \text{Agg}_S^t
$$

$$
\boldsymbol{\theta}_D^{t+1} = \mathbb{I}[t \bmod K = 0]\ \text{Agg}_D^t + \left(1 - \mathbb{I}[t \bmod K = 0]\right) \widehat{\boldsymbol{\theta}}_D^t
$$

$$
\widehat{\boldsymbol{\theta}}_D^{t+1} = \mathbb{I}[t \bmod K = 0]\ \boldsymbol{\theta}_D^{t+1} + \left(1 - \mathbb{I}[t \bmod K = 0]\right) \widehat{\boldsymbol{\theta}}_D^t
$$

If $t \bmod K = 0$ (a *full round*), both shallow and deep partitions are freshly aggregated and the cache is refreshed. Otherwise (a *shallow-only round*), only $\boldsymbol{\theta}_S$ is updated from client uploads and the cached deep parameters are reused.

---

### Communication Cost

Let $|\boldsymbol{\theta}_S|$ and $|\boldsymbol{\theta}_D|$ be the byte sizes of the shallow and deep partitions.

The **uplink communication cost** in round $t$ is

$$
C_{\text{up}}(t) = |\mathcal{P}_t| \left( |\boldsymbol{\theta}_S| + |\boldsymbol{\theta}_D|\ \mathbb{I}[t \bmod K = 0] \right) \quad [\text{bytes}]
$$

The **downlink communication cost** is

$$
C_{\text{down}}(t) = |\mathcal{P}_t| \left( |\boldsymbol{\theta}_S| + |\boldsymbol{\theta}_D| \right) \quad [\text{bytes}]
$$

---

## Dataset (PTB-XL)

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (PhysioNet): large public 12-lead ECG dataset.

- **Task**: Binary classification (e.g. NORM vs abnormal).
- **Splits**: Folds 1–9 train/val, fold 10 test (standard).
- **Signals**: 10 s, 500 Hz, 12 leads → (5000, 12) per recording.
- **Labels**: Diagnostic superclass (NORM, MI, STTC, CD, HYP). Data can be partitioned IID or non-IID across clients for FL.

---

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
├── jobs/
│   ├── run_seed_chain.sh      # One replicate chain: sync FedAvg baseline, then async K (FL_SEED-scoped)
│   ├── run_chains_serial.sh   # Run remaining chains one at a time (memory-bound; never concurrent)
│   ├── run_tonight.sh         # Detached, sleep-guarded launcher for an overnight batch
│   ├── status.sh              # Replicate status: which seeds/regimes are done, accuracy per run
│   ├── validate_run.py        # Check a run's artifacts actually match its metrics file
│   ├── make_results_table.py  # Results table + every number quoted in the paper, from run artifacts
│   ├── precheck_threads.py    # Thread-count determinism and throughput check before a batch
│   ├── watchdog.py            # Detect and recover a stalled (gRPC ping-timeout) chain
│   └── stall_watchdog.sh, start_watchdog.sh, queue_seed13.sh
│
├── paper/
│   ├── main-conference.pdf    # Accepted manuscript (IEEE GLOBECOM 2026), camera-ready in review
│   ├── main-conference.tex    # Paper source
│   ├── ref.bib
│   └── Tikz-Figs/             # pgfplots figures, one .tex per experiment curve
│
├── utils/
│   ├── seed.py                # Seeds Python/NumPy/PyTorch and pins the PyTorch thread count
│   ├── tee_log.py             # Tee stdout/stderr to log file
│   └── ...                    # Config writer, process monitoring, convenience utilities
│
├── LoadData.py                # PTB-XL loader and fold-based splits
├── THESIS.pdf                 # Undergraduate thesis (the long-form version of the paper)
├── requirements.txt
├── .gitignore
└── README.md
```

Results are written to `results/centralized/`, `results/sync-federated/`, and `results/async-federated/` (checkpoints, metrics, plots, logs). Place PTB-XL under `PTB-XL/` at the project root or configure `DATA_PATH` in the configs.

---

## Reproducing the runs

All defaults reproduce the single-seed (seed 42) experiments reported in the thesis. The paper's
multi-seed replicates are driven entirely by environment variables, so the code path is identical
whether or not they are set:

| Variable | Default | Effect |
|---|---|---|
| `FL_SEED` | unset (42) | Seeds Python/NumPy/PyTorch **and** scopes `RUN_ID` and the partition directory (`results/sync-federated_s<seed>/`), so replicates never share or clobber partitions. The async run reads the partitions the sync run of the same seed produced. |
| `FL_K` | `1` | Deep-layer synchronization period K for the async regime. |
| `FL_PORT` | `8080` | Flower server/client port, so two chains can coexist on one host. |
| `FL_NUM_THREADS` | PyTorch default | Pins the PyTorch intra-op thread count. Forward+backward was verified bitwise identical across 1, 4, 8 and 10 threads on this model; pinning makes the reproducibility claim independent of the host. Every run records `TORCH_THREADS` in its `experiment_config.txt`. |

```bash
# Single replicate chain: synchronous FedAvg baseline for seed 7, then async K=2 on the same partitions
FL_SEED=7 FL_K=2 FL_PORT=8081 jobs/run_seed_chain.sh

# Only the async stage (the sync baseline for that seed already exists)
FL_SEED=7 FL_K=2 FL_PORT=8081 FL_STAGES=async jobs/run_seed_chain.sh

# Progress across seeds, and a sanity check that a finished run did what its metrics claim
jobs/status.sh
python jobs/validate_run.py results/async-federated/async_IID_4R_3C_1L_K2_s7 --clients 3 --rounds 4

# Regenerate the paper's results table and quoted figures from the run artifacts
python jobs/make_results_table.py
```

Run chains **serially**. Each chain holds roughly 6.5 GB resident and peaks near 8 GB (three clients
each hold a 1.5 GB partition plus the training graph); two concurrent chains exhausted a 16 GB machine
and panicked the kernel. `jobs/run_chains_serial.sh` and `jobs/run_tonight.sh` encode that constraint.

---

## Experiments


| Experiment | **Architecture** | Data Distribution | Deep-Layer Updates Every K Rounds | Purpose |
|------------|------------------|-------------------|---------------------------|---------|
| C1         | Centralized      | IID               |                           | _Establish_ baseline for model performance. |
| S1         | Synchronous FL   | IID               |                           | _Establish_ baseline FL network communication metrics in **ideal** IoT device data distributions. |
| A1         | Asynchronous FL  | IID               | K=1                       | _Observe_ benefits & tradeoffs of asynchronous FL in **ideal** IoT device data distributions. |
| A2         | Asynchronous FL  | IID               | K=2                       | _Observe_ benefits & tradeoffs of asynchronous FL in **ideal** IoT device data distributions. |
| A3         | Asynchronous FL  | IID               | K=4                       | _Observe_ benefits & tradeoffs of asynchronous FL in **ideal** IoT device data distributions. |
| S2         | Synchronous FL   | non-IID           |                           | _Establish_ baseline FL network communication metrics in **realistic** IoT device data distributions. |
| A1         | Asynchronous FL  | non-IID           | K=2                       | _Observe_ benefits & tradeoffs of asynchronous FL in **realistic** IoT device data distributions. |
| A2         | Asynchronous FL  | non-IID           | K=4                       | _Observe_ benefits & tradeoffs of asynchronous FL in **realistic** IoT device data distributions. |

- **Controls:** Model (ecg_cnn.py), optimizer (Adam), batch size, init partitions, random seed, training epochs, aggregation, evaluation pipeline, 

---

### Dependent Variables (Experimental Outcomes Recorded)

For each experiment, we observe and record the following dependent variables:

**Primary Metrics (Communication/Synchronization Focus)**
- **Total bytes transmitted (system-wide)**
- **Bytes transmitted per client**
- **Number of parameter update messages**
- **Participation-adjusted communication cost**
- **Server-side waiting/straggler metrics (sync cost, async idle time)**
- **Shallow vs. deep parameter update frequency/staleness**

**Secondary Metrics (Model Utility and Stability)**
- **Model convergence (per round/epoch loss)**
- **Predictive performance (accuracy, F1, AUROC as appropriate)**
- **Variation/stability across seeds and experimental repetitions**

All metrics and configurations are logged per run under the appropriate `results/` subdirectory to ensure reproducibility and enable direct controlled comparison for evaluating asynchronous FL methods against baselines.
