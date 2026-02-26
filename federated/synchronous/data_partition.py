"""
Data Partitioning for Federated Learning

Splits training data across multiple clients to simulate distributed IoT devices.
"""

import numpy as np
from typing import List, Tuple
import torch
from torch.utils.data import TensorDataset


def normalize_data(X):
    # Normalize ECG signals to zero mean and unit variance
    mean = X.mean()
    std = X.std()
    return (X - mean) / (std + 1e-8)


def partition_data_iid(X_train, y_train, num_clients: int) -> List[TensorDataset]:
    """
    Partition data in IID (Independent and Identically Distributed) for clients
    Each client gets a random subset of data with similar class distribution
    Used to simulate the IDEAL case where all client devices see similar data
    
    Args:
        X_train: Training ECG signals, shape (n_samples, time_steps, num_leads)
        y_train: Training labels, shape (n_samples,)
        num_clients: Number of clients to partition data across
    
    Returns:
        List of TensorDatasets, one per client
    """
    print(f"\nPartitioning data (IID) across {num_clients} clients...")
    
    X_train = normalize_data(X_train)
    
    n_samples = len(X_train)
    samples_per_client = n_samples // num_clients # samples per client
    
    indices = np.random.permutation(n_samples)
    
    client_datasets = []

    for i in range(num_clients):
        # Get indices for this client
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            # Last client gets remaining samples
            end_idx = n_samples
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_indices = indices[start_idx:end_idx]
        
        X_client = X_train[client_indices]
        y_client = y_train[client_indices]
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_client)
        y_tensor = torch.FloatTensor(y_client).unsqueeze(1)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        client_datasets.append(dataset)
        
        n_positive = y_client.sum()
        n_negative = len(y_client) - n_positive
        print(f"  Client {i+1}: {len(dataset)} samples "
              f"(positive: {int(n_positive)}, negative: {int(n_negative)})")
    
    return client_datasets


def partition_data_non_iid(
    X_train, y_train, num_clients: int, alpha: float = 0.5
) -> List[TensorDataset]:
    """
    Partition data in non-IID manner using a Dirichlet distribution.

    We simulate heterogeneous client distributions by drawing per-class
    proportions from a Dirichlet distribution with concentration alpha.

    For each class c with dataset D_c:
      - Sample p ~ Dir(alpha * 1_N) over N clients
      - Approximately allocate |D_c| * p_j samples of class c to client j

    Args
    ----
    X_train : np.ndarray
        Training ECG signals.
    y_train : np.ndarray
        Training labels.
    num_clients : int
        Number of clients.
    alpha : float, default 0.5
        Dirichlet concentration parameter.
        * Lower alpha → more non-IID (heterogeneous)
        * Higher alpha → more IID (homogeneous)
        * Typical values: 0.1 (very non-IID) to 1.0 (moderately non-IID)

    Returns
    -------
    List[TensorDataset]
        One TensorDataset per client.
    """
    print(f"\nPartitioning data (non-IID, alpha={alpha}) across {num_clients} clients...")
    
    X_train = normalize_data(X_train)
    
    unique_classes = np.unique(y_train) # Get unique classes
    n_classes = len(unique_classes)
    
    client_data = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples to clients using Dirichlet distribution
    for cls in unique_classes:
        # Get indices for this class
        cls_indices = np.where(y_train == cls)[0]
        np.random.shuffle(cls_indices)
        
        # Sample proportions from DIRICHLET DISTRIBUTION
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Split indices according to proportions
        proportions = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
        cls_splits = np.split(cls_indices, proportions)
        
        # Assign to clients
        for client_id, split in enumerate(cls_splits):
            client_data[client_id].extend(split.tolist())
    
    client_datasets = []
    
    for i in range(num_clients):
        client_indices = np.array(client_data[i])
        
        np.random.shuffle(client_indices)
        
        X_client = X_train[client_indices]
        y_client = y_train[client_indices]
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_client)
        y_tensor = torch.FloatTensor(y_client).unsqueeze(1)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        client_datasets.append(dataset)
        
        n_positive = y_client.sum()
        n_negative = len(y_client) - n_positive
        pos_ratio = n_positive / len(y_client) * 100 if len(y_client) > 0 else 0
        print(f"  Client {i+1}: {len(dataset)} samples "
              f"(positive: {int(n_positive)} [{pos_ratio:.1f}%], "
              f"negative: {int(n_negative)})")
    
    return client_datasets


def prepare_federated_data(
    X_train,
    y_train,
    X_test,
    y_test,
    num_clients: int,
    iid: bool = True,
    alpha: float = 0.5,
):
    """
    Prepare data for federated learning.

    Args
    ----
    X_train, y_train : np.ndarray
        Training data and labels.
    X_test, y_test : np.ndarray
        Test data and labels.
    num_clients : int
        Number of clients.
    iid : bool, default True
        If True, create IID partitions; if False, use Dirichlet non-IID.
    alpha : float, default 0.5
        Dirichlet concentration parameter passed to `partition_data_non_iid`
        when `iid` is False.

    Returns
    -------
    client_datasets : List[TensorDataset]
        One training dataset per client.
    test_dataset : TensorDataset
        Shared test dataset for evaluation.
    """
    # Partition training data
    if iid:
        client_datasets = partition_data_iid(X_train, y_train, num_clients)
    else:
        client_datasets = partition_data_non_iid(
            X_train, y_train, num_clients, alpha=alpha
        )
    
    # Prepare test dataset
    X_test_norm = normalize_data(X_test)
    X_test_tensor = torch.FloatTensor(X_test_norm)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    print(f"\nTest set: {len(test_dataset)} samples")
    print("=" * 60)
    
    return client_datasets, test_dataset