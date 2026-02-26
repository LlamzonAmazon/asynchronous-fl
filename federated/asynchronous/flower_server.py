"""
Flower Server for Asynchronous Layer-Wise Federated Learning

This is the core contribution of this thesis, contains:
  - AsyncLayerFedAvg(FedAvg): custom strategy that temporally decouples
    shallow (frequent) and deep (infrequent) aggregation.
  - Server-side helpers ported from sync: evaluate_global_model,
    save_checkpoint, plot_training_curves, save_network_metrics,
    save_run_metadata.

Round type (full / shallow_only) is decided by a pluggable DeepSchedule
object — the strategy itself contains no "if N == …" scheduling logic.

Staleness vs. scheduled partial updates
---------------------------------------

Deep-layer staleness is explicitly tracked on the server via:

  - current_deep_params       : cached deep parameters (θ̂_D^t)
  - deep_last_update_round    : last round in which deep layers were updated
  - deep_staleness_rounds     : derived as (server_round - deep_last_update_round)

However, aggregation itself remains pure FedAvg over the parameters that
are updated in a given round:

  - On full rounds, both shallow and deep partitions are aggregated.
  - On shallow_only rounds, only the shallow partition is aggregated and
    the deep partition is taken from the cache.

No staleness-based reweighting (for example α = 1 / (τ + 1)) is applied to
the updates. In the terminology of the thesis, this implementation is a
scheduled partial update scheme with logged staleness, not a fully
staleness-weighted asynchronous FL method.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Metrics,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from federated.asynchronous.schedule import DeepSchedule


# ======================================================================
# Strategy
# ======================================================================

class AsyncLayerFedAvg(FedAvg):
    """MODIFIED FedAvg with layer-wise update scheduling.

    Downloads are always the full model.  Uploads and aggregation are
    restricted to shallow arrays on shallow_only rounds and cover all
    arrays on full rounds.

    The schedule object (DeepSchedule) is the single source of truth
    for round-type decisions.
    """

    def __init__(
        self,
        schedule: DeepSchedule,
        all_keys: List[str],
        shallow_idxs: List[int],
        deep_idxs: List[int],
        all_keys_hash: str,
        eval_config: Dict,
        config,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Schedule (single source of scheduling truth)
        self.schedule = schedule

        # Layer mapping (single source of index truth)
        self.all_keys: List[str] = all_keys
        self.shallow_idxs: List[int] = shallow_idxs
        self.deep_idxs: List[int] = deep_idxs
        self.all_keys_hash: str = all_keys_hash
        self._shallow_set: set = set(shallow_idxs)
        self._deep_set: set = set(deep_idxs)

        # Deep-layer cache (updated only on full rounds)
        self.current_deep_params: List[np.ndarray] = []
        self.deep_last_update_round: int = 0

        # Round log for communication accounting
        self.round_log: List[Dict] = []

        # History for adaptive schedule
        self.history: Dict[str, List[float]] = {"test_loss": [], "test_accuracy": []}

        # Evaluation helpers (model, test_loader, device, …)
        self.eval_config: Dict = eval_config
        self.fl_config = config

        # Global early stopping (server-side, for analysis/logging)
        self.early_stopping_enabled: bool = getattr(config, "EARLY_STOPPING", False)
        self.early_stopping_patience: int = getattr(config, "PATIENCE", 5)
        self._es_best_loss: Optional[float] = None
        self._es_patience_counter: int = 0
        self._es_early_stopped_round: Optional[int] = None

        # Pre-compute byte sizes from initial parameters (set after
        # initialize_parameters populates current_deep_params).
        self._shallow_bytes: int = 0
        self._deep_bytes: int = 0
        self._full_bytes: int = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Provide initial parameters and cache deep arrays."""
        params = super().initialize_parameters(client_manager)
        if params is not None:
            ndarrays = parameters_to_ndarrays(params)
            self.current_deep_params = [ndarrays[i] for i in self.deep_idxs]
            # Measure byte sizes once
            self._shallow_bytes = sum(
                ndarrays[i].nbytes for i in self.shallow_idxs
            )
            self._deep_bytes = sum(
                ndarrays[i].nbytes for i in self.deep_idxs
            )
            self._full_bytes = self._shallow_bytes + self._deep_bytes
        return params

    # ------------------------------------------------------------------
    # configure_fit  — decides round type, samples clients, builds config
    # ------------------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Build per-client FitIns with round_type + shallow_idxs."""

        round_type = self.schedule.round_type(server_round, self.history)

        # Deterministic client sampling
        sample_size = min(self.fl_config.CLIENTS_PER_ROUND, client_manager.num_available())
        rng = random.Random(self.fl_config.PARTICIPATION_SEED + server_round)
        all_clients = list(client_manager.all().values())
        selected = rng.sample(all_clients, sample_size) if sample_size < len(all_clients) else all_clients

        # Config dict sent to every client
        fit_config: Dict[str, Scalar] = {
            "round_type": round_type,
            "server_round": server_round,
            # Flower Scalar supports str — encode list as JSON string
            "shallow_idxs": json.dumps(self.shallow_idxs),
            "all_len": len(self.all_keys),
            "all_keys_hash": self.all_keys_hash,
        }

        fit_ins = FitIns(parameters, fit_config)
        return [(client, fit_ins) for client in selected]

    # ------------------------------------------------------------------
    # aggregate_fit  — weighted average with partial aggregation
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client results — shallow-only or full."""

        round_type = self.schedule.round_type(server_round, self.history)
        num_selected = len(results) + len(failures)

        # ── Empty results guard ──────────────────────────────────────
        if not results:
            self._log_round(
                server_round=server_round,
                round_type=round_type,
                num_selected=num_selected,
                num_completed=0,
                round_failed=True,
            )
            # Return previous parameters unchanged (evaluation still runs)
            return None, {}

        # ── Extract and validate client arrays ───────────────────────
        weights_results: List[Tuple[List[np.ndarray], int]] = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples

            expected_len = (
                len(self.all_keys) if round_type == "full" else len(self.shallow_idxs)
            )
            if len(ndarrays) != expected_len:
                raise ValueError(
                    f"Client returned {len(ndarrays)} arrays but expected "
                    f"{expected_len} for round_type='{round_type}' "
                    f"(all_keys={len(self.all_keys)}, "
                    f"shallow_idxs={len(self.shallow_idxs)}). "
                    "Aborting — check client serialization."
                )
            weights_results.append((ndarrays, num_examples))

        # ── Weighted average ─────────────────────────────────────────
        aggregated = _weighted_average_arrays(weights_results)

        if round_type == "full":
            # Full FedAvg — update deep cache
            full_params = aggregated
            self.current_deep_params = [aggregated[i] for i in self.deep_idxs]
            self.deep_last_update_round = server_round
        else:
            # Shallow-only: reconstruct full parameter list
            full_params = [None] * len(self.all_keys)
            for pos, idx in enumerate(self.shallow_idxs):
                full_params[idx] = aggregated[pos]
            for pos, idx in enumerate(self.deep_idxs):
                full_params[idx] = self.current_deep_params[pos]

        # ── Communication accounting ─────────────────────────────────
        num_completed = len(results)
        self._log_round(
            server_round=server_round,
            round_type=round_type,
            num_selected=num_selected,
            num_completed=num_completed,
            round_failed=False,
        )

        # ── Return aggregated parameters ─────────────────────────────
        aggregated_metrics: Dict[str, Scalar] = {}
        return ndarrays_to_parameters(full_params), aggregated_metrics

    # ------------------------------------------------------------------
    # evaluate  — server-side evaluation (identical logic to sync)
    # ------------------------------------------------------------------

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on test set, update history, save checkpoint."""
        ndarrays = parameters_to_ndarrays(parameters)
        loss, metrics = evaluate_global_model(server_round, ndarrays, self.eval_config)

        # Feed history for adaptive schedule
        self.history["test_loss"].append(loss)
        self.history["test_accuracy"].append(metrics.get("accuracy", 0.0))

        # Attach eval results to the latest round log entry
        if self.round_log:
            self.round_log[-1]["test_loss"] = loss
            self.round_log[-1]["test_accuracy"] = metrics.get("accuracy", 0.0)

        # Global early stopping (server-side), for analysis/logging only.
        if self.early_stopping_enabled:
            best_loss = self._es_best_loss
            patience_counter = self._es_patience_counter
            early_stopped_round = self._es_early_stopped_round

            if best_loss is None or loss < best_loss:
                best_loss = loss
                patience_counter = 0
                early_stopped_round = None
            else:
                patience_counter += 1
                if early_stopped_round is None and patience_counter >= self.early_stopping_patience:
                    early_stopped_round = server_round
                    print(
                        f"  [Async EarlyStopping] Patience exceeded at round {server_round} "
                        f"(best loss {best_loss:.4f})."
                    )

            self._es_best_loss = best_loss
            self._es_patience_counter = patience_counter
            self._es_early_stopped_round = early_stopped_round

            if self.round_log:
                self.round_log[-1]["early_stopping_best_loss"] = best_loss
                self.round_log[-1]["early_stopping_patience_counter"] = patience_counter
                self.round_log[-1]["early_stopped_round"] = early_stopped_round

        return loss, metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_round(
        self,
        server_round: int,
        round_type: str,
        num_selected: int,
        num_completed: int,
        round_failed: bool,
    ) -> None:
        """Append a communication-accounting entry for this round."""
        deep_staleness = server_round - self.deep_last_update_round

        if round_type == "full":
            bytes_up_per_client = self._full_bytes
        else:
            bytes_up_per_client = self._shallow_bytes

        bytes_up_total = bytes_up_per_client * num_completed
        bytes_down_per_client = self._full_bytes  # always send full model
        bytes_down_total = bytes_down_per_client * num_selected

        upload_fraction_of_full = (
            bytes_up_per_client / self._full_bytes if self._full_bytes > 0 else 0.0
        )

        # Schedule metadata
        sched_desc = self.schedule.description()
        trigger_reason = ""
        if hasattr(self.schedule, "last_trigger_reason"):
            trigger_reason = self.schedule.last_trigger_reason

        entry = {
            "round": server_round,
            "round_type": round_type,
            "round_failed": round_failed,
            "schedule_params": sched_desc,
            "trigger_reason": trigger_reason,
            "num_clients_selected": num_selected,
            "num_clients_completed": num_completed,
            "bytes_up_per_client": int(bytes_up_per_client),
            "bytes_up_total": int(bytes_up_total),
            "bytes_down_per_client": int(bytes_down_per_client),
            "bytes_down_total": int(bytes_down_total),
            "upload_fraction_of_full": upload_fraction_of_full,
            "deep_last_update_round": self.deep_last_update_round,
            "deep_staleness_rounds": deep_staleness,
            # test_loss and test_accuracy filled in by evaluate()
            "test_loss": None,
            "test_accuracy": None,
        }
        self.round_log.append(entry)


# ======================================================================
# Weighted-average helper (mirrors sync FedAvg behaviour)
# ======================================================================

def _weighted_average_arrays(
    results: List[Tuple[List[np.ndarray], int]],
) -> List[np.ndarray]:
    """Compute weighted average of numpy arrays by num_examples."""
    total_examples = sum(n for _, n in results)
    avg = [
        np.zeros_like(results[0][0][i]) for i in range(len(results[0][0]))
    ]
    for ndarrays, num_examples in results:
        weight = num_examples / total_examples
        for i, arr in enumerate(ndarrays):
            avg[i] += arr * weight
    return avg


# ======================================================================
# Server-side evaluation (ported from sync flower_server.py)
# ======================================================================

def evaluate_global_model(
    server_round: int, parameters: List[np.ndarray], config: Dict
) -> Tuple[float, Dict[str, float]]:
    """Evaluate global model on the test set — identical logic to sync."""
    from collections import OrderedDict

    model = config["model"]
    test_loader = config["test_loader"]
    device = config["device"]

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    criterion = nn.BCELoss()
    total_loss = 0.0
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
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    print(
        f"  [Server] Round {server_round}: Test Loss = {avg_loss:.4f}, "
        f"Test Accuracy = {accuracy:.2f}%"
    )

    # Save checkpoint after each round
    if "results_dir" in config:
        save_checkpoint(server_round, model, avg_loss, accuracy, config["results_dir"])
        if "num_rounds" in config and server_round == config["num_rounds"]:
            save_final_model(model, config["results_dir"], config.get("model_save_path"))

    return avg_loss, {"accuracy": accuracy}


# ======================================================================
# Checkpointing (ported from sync)
# ======================================================================

def save_checkpoint(
    round_num: int, model, loss: float, accuracy: float, results_dir: str
) -> None:
    """Save model checkpoint and metrics after each round."""
    checkpoint_dir = Path(results_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / f"model_round_{round_num}.pth"
    torch.save(model.state_dict(), model_path)

    metrics = {"round": round_num, "test_loss": loss, "test_accuracy": accuracy}
    metrics_path = checkpoint_dir / f"metrics_round_{round_num}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Append to cumulative file
    cumulative_path = checkpoint_dir / "all_metrics.json"
    if cumulative_path.exists():
        with open(cumulative_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []
    all_metrics.append(metrics)
    with open(cumulative_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"  [Server] Checkpoint saved: {model_path}")


def save_final_model(
    model, results_dir: str, model_save_path: Optional[str] = None
) -> None:
    """Save the final global model after training completes."""
    if model_save_path:
        final_path = Path(model_save_path)
    else:
        final_path = Path(results_dir) / "global_model.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"  [Server] Final global model saved: {final_path}")


# ======================================================================
# Metric aggregation callback (for client-reported fit metrics)
# ======================================================================

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client metrics via weighted average — identical to sync."""
    total_examples = sum(n for n, _ in metrics)
    aggregated: Dict[str, float] = {}
    if metrics:
        for key in metrics[0][1].keys():
            weighted_sum = sum(n * m[key] for n, m in metrics)
            aggregated[key] = weighted_sum / total_examples
    return aggregated


# ======================================================================
# Post-training outputs
# ======================================================================

def plot_training_curves(
    results_dir: str, plot_save_path: str, round_log: List[Dict]
) -> None:
    """Plot test loss & accuracy with deep/shallow round markers."""
    metrics_path = Path(results_dir) / "checkpoints" / "all_metrics.json"
    if not metrics_path.exists():
        print(f"No metrics at {metrics_path}; skipping training curves.")
        return

    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)
    if not all_metrics:
        print("No round metrics to plot; skipping training curves.")
        return

    rounds = [m["round"] for m in all_metrics]
    test_losses = [m["test_loss"] for m in all_metrics]
    test_accs = [m["test_accuracy"] for m in all_metrics]

    # Identify deep (full) rounds for highlighting
    full_rounds = set()
    for entry in round_log:
        if entry.get("round_type") == "full":
            full_rounds.add(entry["round"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rounds, test_losses, "b-o", label="Test Loss", markersize=4)
    for r in rounds:
        if r in full_rounds:
            ax1.axvline(x=r, color="green", alpha=0.25, linestyle="--", linewidth=1)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss")
    ax1.set_title("Test Loss per Round")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(rounds, test_accs, "r-o", label="Test Accuracy (%)", markersize=4)
    for r in rounds:
        if r in full_rounds:
            ax2.axvline(x=r, color="green", alpha=0.25, linestyle="--", linewidth=1)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Test Accuracy per Round")
    ax2.legend()
    ax2.grid(True)

    # Add legend entry for full-round markers
    from matplotlib.lines import Line2D
    custom_legend = Line2D([0], [0], color="green", alpha=0.5, linestyle="--", label="Full (deep) round")
    for ax in (ax1, ax2):
        handles, labels = ax.get_legend_handles_labels()
        handles.append(custom_legend)
        ax.legend(handles=handles)

    plt.tight_layout()
    Path(plot_save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Training curves saved: {plot_save_path}")


def save_network_metrics(
    round_log: List[Dict],
    results_dir: str,
    config,
    full_model_bytes: int,
    shallow_bytes: int,
    deep_bytes: int,
) -> None:
    """Compute and save aggregate communication metrics + per-round log.

    Definitions (per round t)
    --------------------------
    Let:
      - |θ_S|, |θ_D| be the byte sizes of the shallow and deep partitions,
        such that full_model_bytes = |θ_S| + |θ_D|.
      - P_t be the set of participating (completed) clients in round t.

    Then the logged quantities are:
      - bytes_up_per_client(t)
          = |θ_S| + |θ_D|              if round_type == "full"
          = |θ_S|                      if round_type == "shallow_only"
      - bytes_down_per_client(t)
          = |θ_S| + |θ_D|              (always full model broadcast)
      - bytes_up_total(t)
          = |P_t| * bytes_up_per_client(t)
      - bytes_down_total(t)
          = |P_t| * bytes_down_per_client(t)
      - upload_fraction_of_full(t)
          = bytes_up_per_client(t) / full_model_bytes
          = |θ_S| / (|θ_S| + |θ_D|)    on shallow_only rounds
          = 1.0                        on full rounds

    Aggregate metrics
    -----------------
    Over all rounds, we report:
      - total_bytes_up, total_bytes_down, total_communication_bytes
      - num_deep_rounds, num_shallow_rounds
      - upload_reduction_pct: 1 - (total_bytes_up / total_bytes_up_sync),
        where total_bytes_up_sync assumes full-model uploads every round
        with the same NUM_ROUNDS and CLIENTS_PER_ROUND.
      - mean_upload_fraction: arithmetic mean of upload_fraction_of_full(t)
        over all logged rounds.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # ── Per-round log ────────────────────────────────────────────────
    round_metrics_path = results_path / "round_metrics.json"
    with open(round_metrics_path, "w") as f:
        json.dump(round_log, f, indent=2)
    print(f"Round metrics saved: {round_metrics_path}")

    # ── Aggregate totals ─────────────────────────────────────────────
    total_bytes_up = sum(e["bytes_up_total"] for e in round_log)
    total_bytes_down = sum(e["bytes_down_total"] for e in round_log)
    num_deep_rounds = sum(1 for e in round_log if e["round_type"] == "full")
    num_shallow_rounds = sum(1 for e in round_log if e["round_type"] == "shallow_only")

    upload_fractions = [e.get("upload_fraction_of_full", 0.0) for e in round_log]
    mean_upload_fraction = (
        sum(upload_fractions) / len(upload_fractions) if upload_fractions else 0.0
    )

    # Sync baseline comparison (same participation regime)
    total_bytes_up_sync = (
        config.NUM_ROUNDS * config.CLIENTS_PER_ROUND * full_model_bytes
    )

    bytes_saved = total_bytes_up_sync - total_bytes_up
    reduction_pct = 1.0 - (total_bytes_up / total_bytes_up_sync) if total_bytes_up_sync > 0 else 0.0

    aggregate = {
        "description": (
            "Async layer-wise FL communication metrics. "
            "Download is always full model; upload varies by round type."
        ),
        "full_model_bytes_state_dict": full_model_bytes,
        "shallow_bytes_state_dict": shallow_bytes,
        "deep_bytes_state_dict": deep_bytes,
        "shallow_fraction": shallow_bytes / full_model_bytes if full_model_bytes > 0 else 0.0,
        "num_rounds": config.NUM_ROUNDS,
        "num_deep_rounds": num_deep_rounds,
        "num_shallow_rounds": num_shallow_rounds,
        "clients_per_round": config.CLIENTS_PER_ROUND,
        "total_bytes_up": total_bytes_up,
        "total_bytes_down": total_bytes_down,
        "total_communication_bytes": total_bytes_up + total_bytes_down,
        "total_communication_mb": round((total_bytes_up + total_bytes_down) / (1024 * 1024), 4),
        "total_bytes_up_sync_baseline": total_bytes_up_sync,
        "bytes_saved_vs_sync": bytes_saved,
        "upload_reduction_pct": round(reduction_pct, 6),
        "mean_upload_fraction": round(mean_upload_fraction, 6),
    }

    # Simulated bandwidth (optional)
    if config.SIMULATED_BANDWIDTH_BPS is not None:
        bps = config.SIMULATED_BANDWIDTH_BPS
        aggregate["simulated_bandwidth_bps"] = bps
        aggregate["simulated_transfer_time_per_round_up_s"] = [
            e["bytes_up_total"] * 8 / bps for e in round_log
        ]
        aggregate["total_simulated_upload_time_s"] = sum(
            aggregate["simulated_transfer_time_per_round_up_s"]
        )

    net_path = results_path / "network_metrics.json"
    with open(net_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"Network metrics saved: {net_path}")
    print(f"  Upload reduction vs sync: {reduction_pct * 100:.2f}%")
    print(f"  Total communication: {aggregate['total_communication_mb']:.2f} MB")


def save_run_metadata(
    results_dir: str,
    config,
    all_keys: List[str],
    shallow_idxs: List[int],
    deep_idxs: List[int],
    all_keys_hash: str,
    schedule_desc: Dict,
    full_model_bytes: int,
    shallow_bytes: int,
    deep_bytes: int,
) -> None:
    """Write run_metadata.json documenting every setting."""
    metadata = {
        "experiment": "async_layer_wise_fl",
        "config": {
            "NUM_ROUNDS": config.NUM_ROUNDS,
            "NUM_CLIENTS": config.NUM_CLIENTS,
            "CLIENTS_PER_ROUND": config.CLIENTS_PER_ROUND,
            "LOCAL_EPOCHS": config.LOCAL_EPOCHS,
            "BATCH_SIZE": config.BATCH_SIZE,
            "LEARNING_RATE": config.LEARNING_RATE,
            "DROPOUT_RATE": config.DROPOUT_RATE,
            "DEVICE": config.DEVICE,
            "IID": config.IID,
            "SCHEDULE_TYPE": config.SCHEDULE_TYPE,
            "DEEP_EVERY_N_ROUNDS": config.DEEP_EVERY_N_ROUNDS,
            "WARMUP_ROUNDS": config.WARMUP_ROUNDS,
            "ADAPTIVE_PATIENCE": config.ADAPTIVE_PATIENCE,
            "ADAPTIVE_MIN_GAP": config.ADAPTIVE_MIN_GAP,
            "ADAPTIVE_MAX_GAP": config.ADAPTIVE_MAX_GAP,
            "SHALLOW_PREFIXES": list(config.SHALLOW_PREFIXES),
            "PARTICIPATION_SEED": config.PARTICIPATION_SEED,
            "SIMULATED_BANDWIDTH_BPS": config.SIMULATED_BANDWIDTH_BPS,
            "SYNC_DATA_DIR": config.SYNC_DATA_DIR,
            "RESULTS_DIR": config.RESULTS_DIR,
        },
        "schedule": schedule_desc,
        "model_keys": {
            "ALL_KEYS": all_keys,
            "SHALLOW_IDXS": shallow_idxs,
            "DEEP_IDXS": deep_idxs,
            "all_keys_hash": all_keys_hash,
            "num_all_keys": len(all_keys),
            "num_shallow_keys": len(shallow_idxs),
            "num_deep_keys": len(deep_idxs),
        },
        "byte_sizes": {
            "full_model_bytes_state_dict": full_model_bytes,
            "shallow_bytes_state_dict": shallow_bytes,
            "deep_bytes_state_dict": deep_bytes,
            "shallow_fraction": shallow_bytes / full_model_bytes if full_model_bytes > 0 else 0.0,
        },
    }

    out_path = Path(results_dir) / "run_metadata.json"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Run metadata saved: {out_path}")


# ======================================================================
# Parameter helpers
# ======================================================================

def get_initial_parameters(model) -> Parameters:
    """Get initial model parameters as Flower Parameters object."""
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(ndarrays)


def compute_layer_split(model, config):
    """Compute ALL_KEYS, shallow/deep indices, hash, and byte sizes.

    Returns
    -------
    dict with keys: all_keys, shallow_idxs, deep_idxs, all_keys_hash,
                    shallow_bytes, deep_bytes, full_bytes
    """
    all_keys = list(model.state_dict().keys())
    shallow_idxs = [
        i for i, k in enumerate(all_keys)
        if any(k.startswith(p) for p in config.SHALLOW_PREFIXES)
    ]
    deep_idxs = [i for i in range(len(all_keys)) if i not in set(shallow_idxs)]

    # Deterministic hash (not Python hash() which is randomized per process)
    all_keys_hash = hashlib.sha256("\n".join(all_keys).encode()).hexdigest()

    # Byte sizes from actual arrays
    params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    shallow_bytes = sum(params[i].nbytes for i in shallow_idxs)
    deep_bytes = sum(params[i].nbytes for i in deep_idxs)
    full_bytes = shallow_bytes + deep_bytes

    return {
        "all_keys": all_keys,
        "shallow_idxs": shallow_idxs,
        "deep_idxs": deep_idxs,
        "all_keys_hash": all_keys_hash,
        "shallow_bytes": shallow_bytes,
        "deep_bytes": deep_bytes,
        "full_bytes": full_bytes,
    }
