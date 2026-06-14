#!/bin/bash
#SBATCH --job-name=eval-cross
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-l40-0.grasp.maas
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/teacher_cross_20260613/logs/%j_eval.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/teacher_cross_20260613/logs/%j_eval.err

source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

CHECKPOINT=${1:-runs/wheel-wrench/wheel-wrench/nn/wheel-wrench.pth}

CUDA_VISIBLE_DEVICES=0 python ./isaacgymenvs/train.py headless=True test=True \
  task.env.objSet=cross task=AllegroArmMOAR task.env.axis=z \
  task.env.numEnvs=64 \
  task.env.observationType=full_stack task.env.legacy_obs=True \
  task.env.ablation_mode=no-pc experiment=wheel-wrench \
  train.params.config.user_prefix=wheel-wrench \
  checkpoint="${CHECKPOINT}"

echo "[$(date)] Done."
