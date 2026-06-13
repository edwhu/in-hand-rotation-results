# Teacher Cross — Analysis

## Goal
Train a state-based PPO teacher on the 4-way wheel-wrench task. This checkpoint will be used for both student variants (no-touch and touch+synesthesia).

## Target Metrics (from paper, Table 1)
| Metric | Target |
|---|---|
| CRR (sim, 500 ep) | ~1011 |
| TTF (sim) | ~47.5 s |

## Run Info
| Field | Value |
|---|---|
| SLURM job | 6532935 (dgx009) |
| WandB run | TBD |
| Checkpoint dir | `runs/wheel-wrench/wheel-wrench/` |
| Started | 2026-06-13 |
| Finished | TBD |

## Results
TBD

## Notes
- Teacher does not use cameras (`legacy_obs=True`, `graphics_device_id=-1`)
- 8192 parallel envs, 4000 PPO epochs, ~1 day on a single B200
- Final checkpoint needed for Stage 2 (data collection on GRASP)
