# bc_cross_20260614 — Analysis

## Goal

Compare two sensor ablations of the cross (wheel-wrench) student policy trained via offline BC:
- **prop-only**: proprioception + spin-axis goal (276 dims, no touch, no camera)
- **no-pc (touch)**: proprioception + 16-dim FSR tactile + spin-axis goal (340 dims, no camera)

Hypothesis: touch sensors (FSR tactile) improve over pure proprioception.

## Data

- Teacher: `wheel-wrench-teacher-s1` (cross task, PPO with full privileged state)
- Dataset: `demonstration-cross-s1-multimodplus/` — 409 files × 200 steps × 64 envs = **5.24M transitions**
- Observation structure (85 dims/frame × 4 stacked = 340 total):
  - [0:45]  joint positions + velocities
  - [45:61] FSR tactile (16 dims) ← stripped in prop-only
  - [61:85] spin-axis goal (24 dims)
- prop-only strips tactile → 69 dims/frame × 4 = **276 dims**

## Training

| Run | SLURM Job | Script | Iterations | Status |
|-----|-----------|--------|------------|--------|
| bc-cross-prop-only | 463373 | bc_cross_prop_only.sh | ~13900 | **Cancelled** (converged) |
| bc-cross-no-pc | 463374 | bc_cross_no_pc.sh | ~14050 | **Cancelled** (converged) |

Training code: old per-iteration file I/O (~3.0–3.5s/iter). Cancelled at ~70% after eval showed no improvement from 59% to 70%.

### BC Loss

| Student | Loss at iter ~13345 |
|---------|------|
| prop-only | 0.477 |
| no-pc (touch) | 0.386 |

Touch sensors reduce BC loss by ~19%.

## Evaluation

Eval script: `eval_cross_{prop_only,no_pc}_mid{,2}.sh` — 5000 env steps, `legacy_obs=True`, 64 envs.

### ~59% training (iter ~11800)

| Student | Checkpoint | Episodes | Survived | Mean Ep Reward | Mean Ep Length |
|---------|-----------|----------|----------|----------------|----------------|
| prop-only | model_bc_11750.pth | 1878 | 0 (0%) | 142.75 | 166 / 500 |
| no-pc (touch) | model_bc_11850.pth | 1368 | 0 (0%) | 312.86 | 225 / 500 |

### ~70% training (iter ~13900–14050) — **final checkpoint**

| Student | Checkpoint | Episodes | Survived | Mean Ep Reward | Mean Ep Length |
|---------|-----------|----------|----------|----------------|----------------|
| prop-only | model_bc_13900.pth | 2093 | 0 (0%) | **143.79** | 148.8 / 500 |
| no-pc (touch) | model_bc_14050.pth | 1350 | 0 (0%) | **304.22** | 227.0 / 500 |

## Key Findings

- **Touch (no-pc) gives 2.1× higher episode reward** than prop-only (304 vs 144)
- Both policies eventually drop the object; touch adds ~80 more steps before failure (227 vs 149)
- **Performance plateaued by iter ~11800** — no improvement at 13900 (prop-only: 143→144, no-pc: 313→304)
- Best checkpoints: `model_bc_13900.pth` (prop-only), `model_bc_14050.pth` (no-pc)
