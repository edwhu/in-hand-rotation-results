#!/bin/bash
#SBATCH --job-name=bc-cross-multimodplus
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=23:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-l40-0.grasp.maas
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/teacher_cross_20260613/logs/%j_bc_multimodplus.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/teacher_cross_20260613/logs/%j_bc_multimodplus.err

source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

CUDA_VISIBLE_DEVICES=0 python ./isaacgymenvs/train_distillation.py headless=True \
  task.env.legacy_obs=False distill.bc_training=warmup \
  task.env.objSet=cross task.env.is_distillation=True \
  task=AllegroArmMOAR task.env.numEnvs=64 \
  task.env.observationType=full_stack \
  distill.ablation_mode=multi-modality-plus \
  task.env.ablation_mode=multi-modality-plus \
  distill.teacher_data_dir=demonstration-cross-s1-multimodplus \
  distill.student_logdir=runs/student/bc-cross-multimodplus \
  train.params.config.user_prefix=bc-cross-multimodplus \
  experiment=bc-cross-multimodplus \
  wandb_activate=True wandb_entity=edhu

echo "[$(date)] BC training done."
