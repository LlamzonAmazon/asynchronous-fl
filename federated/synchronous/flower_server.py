"""
Flower Server for Federated Learning

The server:
1. Maintains the global model
2. Sends global model to clients each round
3. Aggregates client updates using FedAvg
4. Evaluates global model on test set
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, ndarrays_to_parameters
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import numpy as np
from pathlib import Path
import json


def evaluate_global_model(server_round: int, parameters, config):
    """
    Evaluate the global model on the test set
    CALLED BY FLOWER after each aggregation round
    Aggregation is the process of combining the model updates from all clients
    
    Args:
        server_round: Current round number
        parameters: Global model parameters (as numpy arrays)
        config: Configuration dictionary
    
    Returns:
        Loss and metrics dictionary
    """

    # Load configs
    model = config["model"]
    test_loader = config["test_loader"]
    device = config["device"]
    
    # Load configs into model
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
    # Evaluate
    model.to(device)
    model.eval()
    
    criterion = nn.BCELoss()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            predictions = (output > 0.5).float()
            correct += (predictions == target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total if total > 0 else 0
    
    print(f"  [Server] Round {server_round}: Test Loss = {avg_loss:.4f}, "
          f"Test Accuracy = {accuracy:.2f}%")
    
    # Save checkpoint after each round
    if "results_dir" in config:
        save_checkpoint(server_round, model, avg_loss, accuracy, config["results_dir"])
        
        # If this is the last round, also save as global_model.pth
        if "num_rounds" in config and server_round == config["num_rounds"]:
            save_final_model(model, config["results_dir"], config.get("model_save_path"))
    
    return avg_loss, {"accuracy": accuracy}


def save_checkpoint(round_num: int, model, loss: float, accuracy: float, results_dir: str):
    """
    Save model checkpoint and metrics after each round
    
    Args:
        round_num: Current round number
        model: PyTorch model to save
        loss: Test loss for this round
        accuracy: Test accuracy for this round
        results_dir: Directory to save checkpoints
    """
    checkpoint_dir = Path(results_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model checkpoint
    model_path = checkpoint_dir / f"model_round_{round_num}.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics = {
        "round": round_num,
        "test_loss": loss,
        "test_accuracy": accuracy
    }
    
    metrics_path = checkpoint_dir / f"metrics_round_{round_num}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also append to a cumulative metrics file
    cumulative_path = checkpoint_dir / "all_metrics.json"
    if cumulative_path.exists():
        with open(cumulative_path, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []
    
    all_metrics.append(metrics)
    
    with open(cumulative_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"  [Server] Checkpoint saved: {model_path}")


def save_final_model(model, results_dir: str, model_save_path: Optional[str] = None):
    """
    Save the final global model after training completes
    
    Args:
        model: PyTorch model to save
        results_dir: Results directory
        model_save_path: Optional custom path for final model
    """
    if model_save_path:
        final_path = Path(model_save_path)
    else:
        final_path = Path(results_dir) / "global_model.pth"
    
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"  [Server] Final global model saved: {final_path}")


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average
    Used to combine evaluation metrics from clients.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
    
    Returns:
        Aggregated metrics dictionary
    """

    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    aggregated = {}
    
    # Calculate weighted average for each metric key 
    if len(metrics) > 0:
        metric_keys = metrics[0][1].keys()
        for key in metric_keys:
            weighted_sum = sum([
                num_examples * m[key] 
                for num_examples, m in metrics
            ])
            aggregated[key] = weighted_sum / total_examples
    
    return aggregated


def create_strategy(model, test_dataset, device, config, initial_parameters):
    """
    Create FedAvg strategy for aggregating client updates
    
    FedAvg (Federated Averaging):
    - Weighted average of client model parameters
    - Weights proportional to number of training samples
    - STANDARD BASELINE FL ALGORITHM
    
    Args:
        model: Global model
        test_dataset: Test dataset for evaluation
        device: Device to run evaluation on
        config: Configuration object
        initial_parameters: Initial model parameters (Flower Parameters object)
    
    Returns:
        Flower strategy object
    """

    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Create evaluation config
    eval_config = {
        "model": model,
        "test_loader": test_loader,
        "device": device,
        "results_dir": config.RESULTS_DIR,
        "num_rounds": config.NUM_ROUNDS,
        "model_save_path": config.MODEL_SAVE_PATH
    }
    
    strategy = FedAvg(
        # Fraction of clients to sample each round
        fraction_fit=1.0,  # 100% of clients (all participate)
        fraction_evaluate=0.0,  # Don't evaluate on clients (only server)
        
        # Minimum number of clients
        # All clients must participate in each round for proper FedAvg
        min_fit_clients=config.NUM_CLIENTS,
        min_evaluate_clients=0,
        min_available_clients=config.NUM_CLIENTS,
        
        # Initial global model parameters - REQUIRED so server doesn't ask client
        initial_parameters=initial_parameters,
        
        # Server-side evaluation
        evaluate_fn=lambda round, params, _: evaluate_global_model(
            round, params, eval_config
        ),
        
        # Aggregate metrics from clients
        fit_metrics_aggregation_fn=weighted_average,
    )
    
    return strategy


def get_initial_parameters(model):
    """
    Get initial model parameters as Flower Parameters object
    
    Args:
        model: PyTorch model
    
    Returns:
        Flower Parameters object containing model parameters
    """
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(ndarrays)