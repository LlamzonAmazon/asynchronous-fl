"""
Pluggable Deep-Layer Update Schedule Module ("Knob Architecture")

All scheduling logic lives here behind one interface:
    schedule.round_type(server_round, history) -> "full" | "shallow_only"

Strategy (AsyncLayerFedAvg) must not contain scheduling logic beyond calling
the schedule object.  Adding a new schedule requires only:
  1. New class in this file
  2. New config fields (if any)
  3. Factory update in create_schedule()

Mathematical view (PeriodicSchedule).

Let:
  - T  = total number of communication rounds (config.NUM_ROUNDS),
  - K  = deep-layer synchronization period (config.DEEP_EVERY_N_ROUNDS),
  - θ  = (θ_S, θ_D) the full model parameters, partitioned into:
       * θ_S : "shallow" parameters — state_dict keys starting with SHALLOW_PREFIXES
       * θ_D : "deep"   parameters — all remaining keys
  - P_t    = set of participating clients in round t,
  - w_i^t  = FedAvg weight for client i in round t
             (proportional to num_examples; sum_i w_i^t = 1),
  - Agg_S^t, Agg_D^t = FedAvg aggregates of shallow / deep parameters
                       after local training in round t,
  - θ̂_D^t = server-side deep cache after round t (current_deep_params),
  - I[·]   = indicator function.

For the periodic schedule, the server update at round t is:

  θ_S^{t+1} = Agg_S^t
  θ_D^{t+1} = I[t mod K = 0] * Agg_D^t + (1 - I[t mod K = 0]) * θ̂_D^t
  θ̂_D^{t+1} = I[t mod K = 0] * θ_D^{t+1} + (1 - I[t mod K = 0]) * θ̂_D^t

so deep layers are only refreshed every K-th (full) round and otherwise
reused from the cache. Shallow parameters are aggregated every round.

Communication cost per round t:

  C_up(t)   = |P_t| * (|θ_S| + |θ_D| * I[t mod K = 0])           [bytes]
  C_down(t) = |P_t| * (|θ_S| + |θ_D|)                           [bytes]

where |θ_S| and |θ_D| are the byte sizes of the shallow and deep partitions
measured from the model state_dict. The implementation in
federated/asynchronous/flower_server.py logs these quantities per round and
in aggregate.
"""

from __future__ import annotations
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class DeepSchedule:
    """Base interface for deep layer update scheduling."""

    def round_type(self, server_round: int, history: Optional[Dict] = None) -> str:
        """Return ``'full'`` or ``'shallow_only'`` for *server_round*."""
        raise NotImplementedError

    def description(self) -> dict:
        """Return schedule config for run_metadata logging."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class PeriodicSchedule(DeepSchedule):
    """Full round every *deep_every_n* rounds; all others are shallow.

    With ``deep_every_n=1`` every round is full — equivalent to sync FedAvg.
    Round 1 is always shallow (unless N=1) because ``1 % N != 0`` for N > 1.
    """

    def __init__(self, deep_every_n: int):
        if deep_every_n < 1:
            raise ValueError(f"deep_every_n must be >= 1, got {deep_every_n}")
        self.deep_every_n = deep_every_n

    def round_type(self, server_round: int, history: Optional[Dict] = None) -> str:
        if self.deep_every_n == 1:
            return "full"
        return "full" if server_round % self.deep_every_n == 0 else "shallow_only"

    def description(self) -> dict:
        return {
            "schedule_type": "periodic",
            "deep_every_n": self.deep_every_n,
        }


class WarmupThenPeriodicSchedule(DeepSchedule):
    """First *warmup_rounds* are all full, then switch to periodic.

    Ensures deep layers get initial convergence before decoupling.
    """

    def __init__(self, warmup_rounds: int, deep_every_n: int):
        if warmup_rounds < 0:
            raise ValueError(f"warmup_rounds must be >= 0, got {warmup_rounds}")
        if deep_every_n < 1:
            raise ValueError(f"deep_every_n must be >= 1, got {deep_every_n}")
        self.warmup_rounds = warmup_rounds
        self.deep_every_n = deep_every_n

    def round_type(self, server_round: int, history: Optional[Dict] = None) -> str:
        if server_round <= self.warmup_rounds:
            return "full"
        if self.deep_every_n == 1:
            return "full"
        return "full" if server_round % self.deep_every_n == 0 else "shallow_only"

    def description(self) -> dict:
        return {
            "schedule_type": "warmup_then_periodic",
            "warmup_rounds": self.warmup_rounds,
            "deep_every_n": self.deep_every_n,
        }


class AdaptivePlateauSchedule(DeepSchedule):
    """Trigger a full round when test loss stagnates.

    * ``patience``  — number of rounds without improvement before triggering full.
    * ``min_gap``   — minimum rounds between consecutive full rounds.
    * ``max_gap``   — maximum rounds between consecutive full rounds.

    **Warm-start rule**: until *patience* rounds of history exist, defaults to
    periodic behavior at *min_gap* cadence so the schedule is well-defined from
    round 1. This rule is implemented inside the class, not as external logic.
    """

    def __init__(self, patience: int, min_gap: int, max_gap: int):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_gap < 1:
            raise ValueError(f"min_gap must be >= 1, got {min_gap}")
        if max_gap < min_gap:
            raise ValueError(f"max_gap ({max_gap}) must be >= min_gap ({min_gap})")
        self.patience = patience
        self.min_gap = min_gap
        self.max_gap = max_gap
        self._last_full_round: int = 0
        self._trigger_reason: str = ""

    def round_type(self, server_round: int, history: Optional[Dict] = None) -> str:
        gap = server_round - self._last_full_round

        # Enforce minimum gap
        if gap < self.min_gap:
            self._trigger_reason = ""
            return "shallow_only"

        # Enforce maximum gap
        if gap >= self.max_gap:
            self._last_full_round = server_round
            self._trigger_reason = "max_gap"
            return "full"

        # Warm-start: not enough history yet → use min_gap cadence
        if history is None or "test_loss" not in history:
            if gap >= self.min_gap:
                self._last_full_round = server_round
                self._trigger_reason = "warmstart_periodic"
                return "full"
            self._trigger_reason = ""
            return "shallow_only"

        losses = history["test_loss"]
        if len(losses) < self.patience:
            # Not enough data yet — periodic at min_gap
            if gap >= self.min_gap:
                self._last_full_round = server_round
                self._trigger_reason = "warmstart_periodic"
                return "full"
            self._trigger_reason = ""
            return "shallow_only"

        # Check for plateau: best loss hasn't improved for `patience` rounds.
        # Find how many rounds since the global-best loss was FIRST achieved —
        # if nothing lower appeared since, the model has stagnated.
        best_loss = min(losses)
        first_best_idx = losses.index(best_loss)
        rounds_since_best = len(losses) - 1 - first_best_idx

        if rounds_since_best >= self.patience:
            self._last_full_round = server_round
            self._trigger_reason = "plateau"
            return "full"

        self._trigger_reason = ""
        return "shallow_only"

    @property
    def last_trigger_reason(self) -> str:
        """Return the reason for the most recent full-round trigger."""
        return self._trigger_reason

    def description(self) -> dict:
        return {
            "schedule_type": "adaptive_plateau",
            "patience": self.patience,
            "min_gap": self.min_gap,
            "max_gap": self.max_gap,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_schedule(config) -> DeepSchedule:
    """Instantiate the schedule class selected in *config*.

    Parameters
    ----------
    config : object
        Must expose ``SCHEDULE_TYPE`` and the relevant schedule-specific fields.

    Returns
    -------
    DeepSchedule
    """
    stype = config.SCHEDULE_TYPE.lower()

    if stype == "periodic":
        return PeriodicSchedule(deep_every_n=config.DEEP_EVERY_N_ROUNDS)

    if stype == "warmup_then_periodic":
        return WarmupThenPeriodicSchedule(
            warmup_rounds=config.WARMUP_ROUNDS,
            deep_every_n=config.DEEP_EVERY_N_ROUNDS,
        )

    if stype == "adaptive_plateau":
        return AdaptivePlateauSchedule(
            patience=config.ADAPTIVE_PATIENCE,
            min_gap=config.ADAPTIVE_MIN_GAP,
            max_gap=config.ADAPTIVE_MAX_GAP,
        )

    raise ValueError(
        f"Unknown SCHEDULE_TYPE '{config.SCHEDULE_TYPE}'. "
        "Choose from: periodic, warmup_then_periodic, adaptive_plateau"
    )
