#!/bin/bash
#SBATCH --job-name=teacher-cross-s2
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=23:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-l40-0.grasp.maas
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/teacher_cross_20260613_s2/logs/%j.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/teacher_cross_20260613_s2/logs/%j.err

source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

CUDA_VISIBLE_DEVICES=0 python ./isaacgymenvs/train.py headless=True seed=2 \
  task.env.objSet=cross task=AllegroArmMOAR task.env.axis=z \
  task.env.numEnvs=8192 train.params.config.minibatch_size=16384 \
  train.params.config.central_value_config.minibatch_size=16384 \
  task.env.observationType=full_stack task.env.legacy_obs=True \
  task.env.ablation_mode=no-pc experiment=wheel-wrench \
  train.params.config.user_prefix=wheel-wrench-s2 wandb_activate=True \
  wandb_entity=edhu

echo "[$(date)] Done."
