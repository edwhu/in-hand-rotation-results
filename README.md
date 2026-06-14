# In-Hand Rotation Results

Companion experiment repository for [`in-hand-rotation`](https://github.com/edwhu/in-hand-rotation), the Robot Synesthesia codebase (ICRA 2024). Each experiment is stored as a dated subfolder with its own scripts, configs, and analysis.

## Research Question

**Does tactile sensing matter for distilled visuotactile policies?**

The paper shows that for the 4-way wheel-wrench task, a student with binary contact + tactile point clouds achieves ~5× higher reward (504 CRR) than a vision-only student (95 CRR). We want to reproduce and extend this comparison, and understand whether binary contact alone drives the benefit or if the full synesthetic point cloud representation is necessary.

## Experiment Plan

All stages run on the **GRASP cluster** (L40/A40 GPUs). Isaac Gym requires older CUDA/driver support that only GRASP provides — the Betty B200 cluster is incompatible.

### Stage 1 — Teacher Training (GRASP / L40)

Train a state-based PPO teacher. The teacher uses full privileged state and does **not** require camera rendering (`legacy_obs=True`).

```
scripts/teacher_cross.sh <GPU_ID>
scripts/teacher_baoding.sh <GPU_ID>
```

Target: ~1011 CRR for cross (paper Table 1). Training takes ~1 day on a single GPU.

### Stage 2 — Data Collection (GRASP / L40)

Roll out the trained teacher to collect transitions. Camera rendering is required (`legacy_obs=False`). A single collection run produces data reusable for all student variants.

```
scripts/collect_cross.sh <GPU_ID> teacher_logdir=<dir> teacher_resume=<checkpoint>
```

Parallelize with multiple `distill.worker_id` values across GPUs.

### Stage 3 — Student BC Training (GRASP / L40)

Train student variants from the collected dataset. All use the same data; only `ablation_mode` differs.

| Student | `ablation_mode` | Input |
|---|---|---|
| No-touch | `no-tactile` | Proprioception + Camera PC + Augmented PC |
| Touch (binary only) | `no-pc` | Proprioception + Binary contact (FSR tactile) |
| Touch + Synesthesia | `multi-modality-plus` | Proprioception + Binary contact + Camera PC + Augmented PC + Tactile PC |
| Prop-only | `prop-only` | Proprioception + task goal only (no touch, no camera) |

---

## Folder Index

| Folder | Date | Status | Purpose |
|---|---|---|---|
| `teacher_cross_20260613` | 2026-06-13 | Done | Cross (wheel-wrench) teacher S1 — `wheel-wrench-teacher-s1` |
| `teacher_cross_20260613_s2` | 2026-06-13 | Done | Cross teacher S2 attempt (unused) |
| `teacher_baoding_20260613` | 2026-06-13 | Done | Baoding teacher S1 — `baoding-teacher-s1` |
| `teacher_baoding_20260613_s2` | 2026-06-13 | Done | Baoding teacher S2 attempt (unused) |
| `bc_cross_20260614` | 2026-06-14 | **Done** | BC students (prop-only, no-pc) on cross task — [analysis](bc_cross_20260614/analysis.md) |
| `bc_baoding_20260614` | 2026-06-14 | **Running** | BC students (prop-only, no-pc) on baoding task — [analysis](bc_baoding_20260614/analysis.md) |

## Results (2026-06-14)

### Cross task — converged at iter ~13900 (training cancelled, no improvement after ~11800)

| Student | Mean Ep Reward | Mean Ep Length | vs prop-only |
|---------|---------------|----------------|--------------|
| prop-only | 143.79 | 148.8 / 500 | — |
| no-pc / touch | 304.22 | 227.0 / 500 | **+2.1×** |

Both eventually drop the object. Touch adds ~80 more steps before failure. Neither survives to 500 steps.

### Baoding task — early eval at iter 3250 (16% trained, final eval pending)

| Student | Mean Ep Reward | Mean Ep Length | vs prop-only |
|---------|---------------|----------------|--------------|
| prop-only | 268.04 | 116.5 / 500 | — |
| no-pc / touch | 301.59 | 127.6 / 500 | **+1.13×** |

Gap expected to grow as training continues.

### BC Loss (touch consistently lower across both tasks)

| Task | prop-only | no-pc (touch) | Δ |
|------|-----------|---------------|---|
| Cross (iter ~13900) | 0.477 | 0.386 | −19% |
| Baoding (iter ~675) | 0.386 | 0.324 | −16% |

## Key Reference Numbers (from paper)

| Condition | CRR (sim) | TTF (sim) |
|---|---|---|
| Teacher (full state) | 1011 | 47.5 s |
| Student: No-touch (Cam+Aug) | 95 | 15.2 s |
| Student: Touch only | 363 | 23.6 s |
| Student: Touch+Cam+Aug+Syn | 504 | 29.2 s |
