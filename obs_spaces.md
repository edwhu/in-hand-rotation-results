# Observation Spaces

Reference for teacher and student observation spaces across tasks. All observations use `n_stack=4` (4-frame history). Computed in `isaacgymenvs/tasks/allegro_arm_morb_axis.py`.

## Per-frame layout (85 dims, shared across tasks)

| Slice | Dims | Content |
|-------|------|---------|
| [0:22] | 22 | Joint positions — 6 arm (zeroed) + 16 hand |
| [22:45] | 23 | Joint velocities — arm zeroed, hand replaced with previous joint targets |
| [45:61] | 16 | FSR tactile contact sensors (binary-ish, 16 fingertip sensors) |
| [61:85] | 24 | Spin axis goal — 3D rotation axis repeated 8× for temporal context |

## Teacher vs. student obs

The teacher receives the full stacked obs **plus** an object buffer appended at the end. The student buffer is assigned **before** the object buffer is concatenated, so the student never directly observes object pose/velocity.

| Buffer | Who | Dims | Formula |
|--------|-----|------|---------|
| `obs_buf` (teacher) | PPO teacher | 353 / 366 | 85×4 + obj_buf |
| `student_obs_buf` | Student BC | 340 | 85×4 (no obj_buf) |

### Object buffer

| Task | Dims | Content |
|------|------|---------|
| Cross / wheel-wrench | 13 | Position (3) + quaternion (4) + linear vel (3) + angular vel (3) |
| Baoding | 26 | Same × 2 objects |

## Total dimensions by obs type

| `observationType` | Teacher | Student |
|-------------------|---------|---------|
| `full_stack` (cross) | **353** | **340** |
| `full_stack_baoding` | **366** | **340** |

## Student ablation modes (applied in `distill_bc_warmup.py`)

Ablation filtering is done on the collected 340-dim student obs, not in the env.

| `ablation_mode` | Dims | What's stripped |
|-----------------|------|-----------------|
| `no-pc` (touch only) | **340** | Nothing — full student obs |
| `prop-only` | **276** | Tactile [45:61] stripped each frame → 69×4 |
| `no-tactile` (camera) | **276** | Same strip, but pointcloud added separately |
| `multi-modality-plus` | **340** + PC | Full obs + camera pointcloud |

## Notes

- `legacy_obs=True` skips camera rendering but does **not** change any observation dimensions. Safe to use for `no-pc` and `prop-only` students at eval time.
- Arm joints [0:6] and arm velocities are always zeroed in the obs — only the 16 hand joints carry information.
- The spin axis goal uses 24 dims (3D axis × 8) rather than 3 to give the network redundant temporal signal.
