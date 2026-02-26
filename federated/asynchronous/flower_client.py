"""
Flower Client for Asynchronous Layer-Wise Federated Learning

Mirrors federated/synchronous/flower_client.py.  Key difference:
  - Client receives ``shallow_idxs`` from the server each round via
    ``FitIns.config`` (single source of truth).
  - On ``shallow_only`` rounds the client returns only the shallow arrays.
  - On ``full`` rounds the client returns all arrays (same as sync).

Local training is always full-model — round type only controls what is
uploaded (not what is trained).
"""

import hashlib
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

import flwr as fl


class AsyncECGClient(fl.client.NumPyClient):
    """Flower client for async layer-wise ECG federated learning.

    Constructor does NOT take ``shallow_idxs`` — the client receives
    indices from the server each round via ``FitIns.config``, ensuring a
    single source of truth.
    """

    def __init__(
        self,
        client_id: int,
        model,
        trainset,
        device: str,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        local_epochs: int = 1,
    ):
        self.client_id = client_id
        self.model = model
        self.trainset = trainset
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs

        self.trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------
    # Parameter serialization (identical to sync)
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return all model state_dict entries as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load full model from numpy arrays — identical to sync."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # fit — local training + partial/full upload
    # ------------------------------------------------------------------

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train full model locally; upload shallow or full arrays."""

        # 1. Load global parameters (full model — same as sync)
        self.set_parameters(parameters)

        # 2. Sanity-check key alignment with server
        local_keys = list(self.model.state_dict().keys())
        expected_len = int(config["all_len"])
        expected_hash = str(config["all_keys_hash"])

        if len(local_keys) != expected_len:
            raise RuntimeError(
                f"[Client {self.client_id}] Key count mismatch: "
                f"local={len(local_keys)}, server={expected_len}"
            )

        local_hash = hashlib.sha256("\n".join(local_keys).encode()).hexdigest()
        if local_hash != expected_hash:
            raise RuntimeError(
                f"[Client {self.client_id}] Key hash mismatch: "
                f"local={local_hash[:12]}…, server={expected_hash[:12]}…  "
                "Model architecture differs between client and server."
            )

        # 3. Train locally for local_epochs — identical to sync
        self.model.to(self.device)
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        print(
            f"\n[Client {self.client_id}] Training on {len(self.trainset)} samples.",
            flush=True,
        )

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

                if (batch_idx + 1) % 20 == 0:
                    print(
                        f"[Client {self.client_id}] Epoch {epoch + 1}/{self.local_epochs}, "
                        f"batch {batch_idx + 1}/{len(self.trainloader)}, "
                        f"loss: {loss.item():.4f}",
                        flush=True,
                    )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(
            f"\n[Client {self.client_id}] Done. Average loss: {avg_loss:.4f}\n",
            flush=True,
        )

        # 4. Decide what to upload based on round_type
        round_type = config.get("round_type", "full")
        shallow_idxs: List[int] = json.loads(str(config["shallow_idxs"]))
        all_params = self.get_parameters(config={})

        if round_type == "shallow_only":
            upload = [all_params[i] for i in shallow_idxs]
            sent_label = "shallow"
        else:
            upload = all_params
            sent_label = "full"

        return (
            upload,
            len(self.trainset),
            {"loss": avg_loss, "sent": sent_label, "num_arrays": len(upload)},
        )

    # ------------------------------------------------------------------
    # evaluate — identical to sync
    # ------------------------------------------------------------------

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate model on local data — identical to sync."""
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                predictions = (output > 0.5).float()
                correct += (predictions == target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(self.trainloader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return avg_loss, len(self.trainset), {"accuracy": accuracy}


# ======================================================================
# Client factory (mirrors sync create_client_fn)
# ======================================================================

def create_client_fn(client_datasets, model_class, device, config):
    """Factory function to create async client instances."""

    def client_fn(cid: str) -> AsyncECGClient:
        client_id = int(cid)
        model = model_class(
            num_leads=config.NUM_LEADS,
            dropout_rate=config.DROPOUT_RATE,
        )
        trainset = client_datasets[client_id]
        return AsyncECGClient(
            client_id=client_id,
            model=model,
            trainset=trainset,
            device=device,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            local_epochs=config.LOCAL_EPOCHS,
        )

    return client_fn
