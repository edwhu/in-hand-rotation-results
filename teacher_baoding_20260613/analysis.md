# Teacher Baoding — Analysis

## Goal
Train a state-based PPO teacher on the two-ball baoding rotation task. This checkpoint will be used for both student variants (no-touch and touch+synesthesia). Baoding is the most contact-sensitive task — maintaining rolling contact on two balls simultaneously makes tactile sensing especially valuable.

## Target Metrics (from paper, Table 1)
| Metric | Target |
|---|---|
| CRR (sim, 500 ep) | TBD from paper |
| TTF (sim) | TBD from paper |

## Run Info
| Field | Value |
|---|---|
| SLURM job | TBD |
| WandB run | TBD |
| Checkpoint dir | `runs/baoding/baoding/` |
| Started | 2026-06-13 |
| Finished | TBD |

## Results
TBD

## Notes
- Teacher does not use cameras (`legacy_obs=True`, `graphics_device_id=-1`)
- 8192 parallel envs, same PPO config as cross teacher
- `observationType=full_stack_baoding` (baoding-specific observation with two-ball state)
- Full run (~20100 epochs) needs ~42h; wall time is 23.5h → will need checkpoint resume
