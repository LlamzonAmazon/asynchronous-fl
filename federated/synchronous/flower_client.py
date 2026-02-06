"""
Flower Client for Federated Learning

Each client represents a healthcare IoT device that:
1. Receives the global model from the server
2. Trains locally on its own data
3. Sends model parameter UPDATES back to the server
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple
from collections import OrderedDict


class ECGClient(fl.client.NumPyClient):
    """
    Flower client for ECG classification
    
    This client trains a local copy of the global model on its private data.
    It never shares raw data, only model parameters.
    """
    
    def __init__(self, client_id: int, model, trainset, device: str, 
                 batch_size: int = 32, learning_rate: float = 0.001, 
                 local_epochs: int = 1):
        """
        Initialize the client
        
        Args:
            client_id: Unique identifier for this client
            model: Neural network model (NOTE: same architecture as global model)
            trainset: This client's private training data
            device: 'mps' or 'cpu'
            batch_size: Batch size for local training
            learning_rate: Learning rate for local optimizer
            local_epochs: Number of epochs to train locally per round
        """
        self.client_id = client_id
        self.model = model
        self.trainset = trainset
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        
        # Create PyTorch DataLoader for local training
        self.trainloader = DataLoader(
            trainset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        # Set loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters as a list of numpy arrays
        
        Flower's NumPyClient expects this to return raw numpy arrays.
        Flower internally wraps them in a Parameters object for gRPC serialization.
        
        Args:
            config: Configuration dictionary (unused here)
        
        Returns:
            List of numpy arrays containing model parameters
        """
        # Convert PyTorch tensors to numpy arrays
        # model.state_dict() returns a dictionary of model parameters (weights & biases)
        # Example: {'conv1.weight': tensor([[...]]), 'conv1.bias': tensor([...])}
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from numpy arrays
        
        This is called by Flower to update the client's model with
        the global model parameters from the server.
        
        Args:
            parameters: List of numpy arrays containing model parameters
        """

        # Load into model
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data
        MAIN TRAINING FUNCTION – CALLED IN EACH FL ROUND
        
        Args:
            parameters: Global model parameters from server
            config: Configuration for this round
        
        Returns:
            Updated model parameters, number of samples, metrics dictionary
        """

        # Update local model with global parameters
        self.set_parameters(parameters)
        
        # Set training parameters
        self.model.to(self.device)
        self.model.train()

        total_loss = 0
        num_batches = 0
        
        print(f"\n[Client {self.client_id}] Training on {len(self.trainset)} samples.", flush=True)
        
        # ==========================================================================================
        # Train for local_epochs (CURRENTLY SET TO 1 EPOCH PER ROUND)
        # ==========================================================================================
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Print progress every 20 batches
                if (batch_idx + 1) % 20 == 0:
                    print(f"[Client {self.client_id}] Epoch {epoch+1}/{self.local_epochs}, "
                          f"batch {batch_idx+1}/{len(self.trainloader)}, loss: {loss.item():.4f}", flush=True)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        print(f"\n[Client {self.client_id}] Done. Average loss: {avg_loss:.4f}\n", flush=True)
        
        # Return updated parameters and metrics
        # NumPyClient.fit() expects a list of NUMPY ARRAYS, not a Parameters object
        return (
            self.get_parameters(config={}),
            len(self.trainset),
            {"loss": avg_loss}
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local data
        
        Args:
            parameters: Model parameters to evaluate
            config: Configuration dictionary
        
        Returns:
            Loss, number of samples, metrics dictionary
        """

        # Update model with given parameters
        self.set_parameters(parameters)
        
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0
        
        # ==========================================================================================
        # Evaluate the model
        # ==========================================================================================
        with torch.no_grad(): # No need to track gradients for evaluation
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                predictions = (output > 0.5).float()
                correct += (predictions == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.trainloader)
        accuracy = 100 * correct / total if total > 0 else 0
        
        return avg_loss, len(self.trainset), {"accuracy": accuracy}


def create_client_fn(client_datasets, model_class, device, config):
    """
    Factory function to create client instances
    Used by flower to create clients
    
    Args:
        client_datasets: List of datasets, one per client
        model_class: Model class to instantiate
        device: Device to train on
        config: Configuration object
    
    Returns:
        Function that creates a client given a client_id
    """
    def client_fn(cid: str) -> ECGClient:
        """Create a client for the given client ID"""
        client_id = int(cid)
        
        # Create a fresh model for this client
        model = model_class(
            num_leads=config.NUM_LEADS,
            dropout_rate=config.DROPOUT_RATE
        )
        
        trainset = client_datasets[client_id]
        
        return ECGClient(
            client_id=client_id,
            model=model,
            trainset=trainset,
            device=device,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            local_epochs=config.LOCAL_EPOCHS
        )
    
    return client_fn