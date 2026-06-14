#!/bin/bash
#SBATCH --job-name=bench-a40
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-a40-1.grasp.maas
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/benchmark_rendering_20260613/logs/%j_bench_a40.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/benchmark_rendering_20260613/logs/%j_bench_a40.err

source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

COMMON="headless=True task=AllegroArmMOAR task.env.objSet=cross \
  task.env.legacy_obs=False task.env.observationType=full_stack \
  task.env.is_distillation=True task.env.ablation_mode=multi-modality-plus \
  distill.ablation_mode=multi-modality-plus distill.bc_training=collect \
  distill.teacher_data_dir=benchmark-tmp-a40 \
  distill.teacher_logdir=runs distill.teacher_resume=wheel-wrench-teacher-s1 \
  train.params.config.user_prefix=bench-a40 wandb_activate=False experiment=bench"

echo "=== [$(date)] numEnvs=64 ==="
CUDA_VISIBLE_DEVICES=0 python ./isaacgymenvs/benchmark_rendering.py ${COMMON} task.env.numEnvs=64

echo "=== [$(date)] numEnvs=256 ==="
CUDA_VISIBLE_DEVICES=0 python ./isaacgymenvs/benchmark_rendering.py ${COMMON} task.env.numEnvs=256

echo "[$(date)] Done."
