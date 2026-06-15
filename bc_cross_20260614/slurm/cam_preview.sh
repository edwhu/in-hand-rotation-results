#!/bin/bash
#SBATCH --job-name=cam-preview
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-l40-0.grasp.maas
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_cross_20260614/logs/%j_cam_preview.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_cross_20260614/logs/%j_cam_preview.err

export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}

source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

echo "[$(date)] Camera preview starting"

PREVIEW_DIR=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_cross_20260614/cam_preview

COMMON="headless=True task=AllegroArmMOAR task.env.objSet=cross \
  task.env.legacy_obs=True task.env.observationType=full_stack \
  task.env.is_distillation=True task.env.numEnvs=1 \
  task.env.record_video=True \
  distill.bc_training=cam_preview distill.ablation_mode=no-pc \
  distill.video_path=${PREVIEW_DIR} \
  distill.teacher_data_dir=demonstration-cross-s1-multimodplus \
  distill.teacher_logdir=runs distill.teacher_resume=wheel-wrench-teacher-s1 \
  distill.student_logdir=runs/student/bc-cross-no-pc \
  distill.student_resume=runs/student/bc-cross-no-pc/model_bc_14050.pth \
  train.params.config.minibatch_size=1024 \
  train.params.config.central_value_config.minibatch_size=1024 \
  train.params.config.user_prefix=cam-preview \
  experiment=cam-preview wandb_activate=False"

CUDA_VISIBLE_DEVICES=0 python -u ./isaacgymenvs/train_distillation.py ${COMMON}

echo "[$(date)] Camera preview done. PNGs at ${PREVIEW_DIR}"
