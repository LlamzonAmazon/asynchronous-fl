# Asychronous Weight-Updating Federated Learning

This is the codebase for my thesis, titled: 

***Applying Asynchronous Weight-Updating in Federated Learning Networks for IoT Health Devices***

Traditional Federated Learning neural networks are slow and inefficient because clients naively send parameter updates synchronously to the global model. Clients send shallow-layer and deep-layer updates in every training round, which results in unnecessary network overhead. The drawbacks of this inefficiency is especially prevalent in healthcare IoT devices with limited computational capacity, and must devote the most computational power possible towards detecting health anomalies.

This project aims to reduce network overhead and computational strain on healthcare IoT devices by implementing the ***asynchronous weight updating*** network communication scheduling strategy. Shallow and deep layer parameter update sending is *temporally decoupled*, and shallow parameter updates are sent more frequently than deep parameter updates. This endeavour also focuses on maintaining model accuracy under healthcare IoT device constaints such as network bandwidth, computational power, et cetera.

## Data
This project uses [*PTB-XL, a large publicly available electrocardiography dataset*](https://physionet.org/content/ptb-xl/1.0.3/) from *PhysioNet*.

## Modules
- PyTorch
- NumPy
- Pandas
- Matplotlib

## Structure
```
asynchronous-fl/
├── .vscode/                        # configs for code debugging 
├── centralized/
│   ├── config.py                   # training configurations
│   └── train.py                    # trains ECG CNN model on PTB-XL 
│
├── experiments/
│   ├── TBD
│
├── federated/
│   ├── TBD
│
├── models/
│   ├── ecg_cnn.py                  # CNN model on PTB-XL (centralized baseline)
│
├── PTB-XL/                         # PTB-XL dataset from PhysioNet (.gitignore'd)
│
├── results/                        # model results (.gitignore'd)
│   ├── async-federated/            # To compare to baseline results
│   ├── centralized/                # Baseline
│   └── sync-federated/             # Baseline
│
├── venv/                           # virtual environment (.gitignore'd)
├── .gitignore
├── LoadData.py
├── REAMDE.md
├── requirements.txt                # dependencies
```

## Author
**Thomas Llamzon** – Honours Computer Science at Western University
