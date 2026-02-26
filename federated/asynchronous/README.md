# Module Structure

## 1. `config.py` — Settings
- Mirrors sync config identically for model/training/data settings. 
- Adds async knobs: `SCHEDULE_TYPE`, `DEEP_EVERY_N_ROUNDS`, `WARMUP_ROUNDS`, `ADAPTIVE_PATIENCE`/`MIN_GAP`/`MAX_GAP`, `SHALLOW_PREFIXES`, `PARTICIPATION_SEED`, `SIMULATED_BANDWIDTH_BPS`. 
- Training budget parity: `NUM_ROUNDS=15`, `LOCAL_EPOCHS=1`.

## 2. `schedule.py` — Deep-Layer Sending Scheduling Schemes
- Three schedule classes behind one `DeepSchedule` interface:
  1. `PeriodicSchedule` — full every N rounds; N=1 = sync equivalent
  2. `WarmupThenPeriodicSchedule` — all-full warmup then periodic
  3. `AdaptivePlateauSchedule` — triggers full on test-loss stagnation with min/max gap bounds
- Factory `create_schedule(config)` reads `SCHEDULE_TYPE` and instantiates. 
- Adding a new schedule only requires a new class + factory entry.

## 3. `flower_server.py` — Asynchronous FedAvg
- `configure_fit`: queries schedule for round type, deterministic client sampling via seed, sends `shallow_idxs`, `all_keys_hash`, `all_len` to clients
- `aggregate_fit`: validates array counts, weighted-average aggregation (full or shallow-only with deep cache), empty-results guard (keeps previous params, logs `round_failed=True`)
- `evaluate`: server-side eval identical to sync, feeds history for adaptive schedule
- **Staleness tracking (no staleness weighting)**:
  - The server maintains a deep-parameter cache (`current_deep_params`) and logs when it was last refreshed (`deep_last_update_round`).
  - Per-round staleness in units of rounds is recorded as `deep_staleness_rounds`.
  - Aggregation itself remains pure FedAvg on the parameters that are updated in that round: full FedAvg on full rounds; shallow-only FedAvg plus cached deep parameters on shallow-only rounds.
  - In the thesis terminology this is a scheduled partial update scheme with tracked staleness, not a fully staleness-weighted async FL method.
- **Communication accounting**: per-round bytes up/down, staleness tracking, schedule metadata
- **Post-training**: `save_run_metadata()`, `save_network_metrics()` (with sync baseline comparison + `upload_reduction_pct`), `plot_training_curves()` (with full-round vertical markers)

## 4. `flower_client.py` — Asynchronous Client
- Receives `shallow_idxs` from server each round (no constructor indices)
- Sanity-checks `all_len + all_keys_hash` — mismatch raises exception
- Trains full model every round (round type only controls upload)
- Serialization matches sync exactly: `state_dict().items()` ordering

## 5. `run_fl.py` — Module Orchestrator
Mirrors sync orchestrator. Verifies sync partition artifacts exist (raises clear error if missing). Prints async-specific config summary.

## 6. `start_server.py` — Async Server launcher
Computes layer split, creates schedule, instantiates AsyncLayerFedAvg, saves run metadata before training, saves network metrics + plots after.

## 7. `start_client.py` — Async Client launcher
Loads partition data from `SYNC_DATA_DIR`, creates `AsyncECGClient`.

---

## Usage

- Ensure the synchronous FL baseline has been run at least once so that
  shared partition artifacts exist in `results/sync-federated/`:
  - `python federated/synchronous/run_fl.py`

- Then run the async orchestrator (reuses the same partitions and writes
  async-specific metrics to `results/async-federated/`):
  - `python federated/asynchronous/run_fl.py`
