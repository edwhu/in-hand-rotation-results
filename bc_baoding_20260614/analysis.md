# bc_baoding_20260614 — Analysis

## Goal

Same prop-only vs no-pc (touch) ablation as `bc_cross_20260614`, but on the **baoding** task
(double-ball rotation, `objSet=ball`, `observationType=full_stack_baoding`).

## Data

- Teacher: `baoding-teacher-s1` (baoding PPO teacher, full privileged state)
- Dataset: `demonstration-baoding-s1-multimodplus/` — 1096 files × 200 steps × 64 envs = **14.03M transitions**
- Observation structure (85 dims/frame × 4 stacked = 340 total):
  - Same layout as cross; prop-only strips tactile → 276 dims

## Training

| Run | SLURM Job | Script | Iterations | Status |
|-----|-----------|--------|------------|--------|
| bc-baoding-prop-only | 463421 | bc_baoding_prop_only.sh | 681/20000 (3.4%) | **Running** (~4.9h left) |
| bc-baoding-no-pc | 463422 | bc_baoding_no_pc.sh | 670/20000 (3.4%) | **Running** (~4.9h left) |

Training code: GPU preload (14.03M samples on L40 48GB VRAM) + `torch.randint` sampling → **~0.92s/iter**.

### BC Loss (at iter ~675)

| Student | Loss |
|---------|------|
| prop-only | 0.386 |
| no-pc (touch) | 0.324 |

Touch sensors reduce BC loss by ~16%, consistent with cross task finding.

## Evaluation

### ~16% training (iter 3250) — early checkpoint

| Student | Checkpoint | Episodes | Survived | Mean Ep Reward | Mean Ep Length |
|---------|-----------|----------|----------|----------------|----------------|
| prop-only | model_bc_3250.pth | 2650 | 0 (0%) | **268.04** | 116.5 / 500 |
| no-pc (touch) | model_bc_3250.pth | 2406 | 0 (0%) | **301.59** | 127.6 / 500 |

Touch gives **1.13×** reward at 16% training — gap expected to widen. Baoding per-step reward (~2.3)
is much higher than cross (~1.0–1.4) due to different reward scales.

### Final eval (model_bc_20000.pth)

Eval scripts ready at `slurm/eval_baoding_{prop_only,no_pc}.sh`.

- [ ] Submit once training finishes (~midnight UTC 2026-06-14)

## Next Steps

- [ ] Submit final evals when training completes
- [ ] Compare prop-only vs no-pc on baoding (expect gap to grow beyond 1.13×)
- [ ] Compare cross vs baoding: cross shows 2.1× touch benefit at convergence; does baoding match?
