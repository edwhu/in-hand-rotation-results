# In-Hand Rotation Results

Companion experiment repository for [`in-hand-rotation`](https://github.com/edwhu/in-hand-rotation), the Robot Synesthesia codebase (ICRA 2024). Each experiment is stored as a dated subfolder with its own scripts, configs, and analysis.

## Research Question

**Does tactile sensing matter for distilled visuotactile policies?**

The paper shows that for the 4-way wheel-wrench task, a student with binary contact + tactile point clouds achieves ~5× higher reward (504 CRR) than a vision-only student (95 CRR). We want to reproduce and extend this comparison, and understand whether binary contact alone drives the benefit or if the full synesthetic point cloud representation is necessary.

## Experiment Plan

### Stage 1 — Teacher Training (Betty / B200)

Train a state-based PPO teacher on the **4-way wheel-wrench** task. The teacher observes full privileged state (joint positions, binary contact, object pose/velocity, shape embedding) and does **not** require camera rendering, making it suitable for the B200 cluster.

```
scripts/teacher_cross.sh <GPU_ID>
```

Target: ~1011 CRR (paper Table 1). Training takes ~1 day on a single GPU with 8192 parallel envs.

### Stage 2 — Data Collection (GRASP / L40)

Roll out the trained teacher to collect 5 M transitions. Camera rendering is required (`legacy_obs=False`). A single collection run produces data reusable for all student variants.

```
scripts/collect_cross.sh <GPU_ID> teacher_logdir=<dir> teacher_resume=<checkpoint>
```

Parallelize with multiple `distill.worker_id` values across GPUs.

### Stage 3 — Student BC Training (GRASP / L40)

Train the following student variants from the collected dataset. All use the same data; only `ablation_mode` differs.

| Student | `ablation_mode` | Input |
|---|---|---|
| No-touch | `no-tactile` | Proprioception + Camera PC + Augmented PC |
| Touch (binary only) | `no-pc` | Proprioception + Binary contact |
| Touch + Synesthesia | `multi-modality-plus` | Proprioception + Binary contact + Camera PC + Augmented PC + Tactile PC |

See `plan_touch_distillation.md` in the `in-hand-rotation` repo for the exact commands.

---

## Folder Index

| Folder | Date | Status | Purpose |
|---|---|---|---|
| *(none yet)* | | | |

## Key Reference Numbers (from paper)

| Condition | CRR (sim) | TTF (sim) |
|---|---|---|
| Teacher (full state) | 1011 | 47.5 s |
| Student: No-touch (Cam+Aug) | 95 | 15.2 s |
| Student: Touch only | 363 | 23.6 s |
| Student: Touch+Cam+Aug+Syn | 504 | 29.2 s |
