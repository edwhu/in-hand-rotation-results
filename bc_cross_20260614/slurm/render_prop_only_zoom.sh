#!/bin/bash
#SBATCH --job-name=render-cross-prop-only
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_cross_20260614/logs/render_prop_only_zoom_%j.log

cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation
source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate

CUDA_VISIBLE_DEVICES=0 python ./isaacgymenvs/train_distillation.py \
  headless=True task.env.record_video=True \
  task.env.legacy_obs=False task.env.is_distillation=True \
  task.env.objSet=cross task=AllegroArmMOAR \
  task.env.numEnvs=4 \
  task.env.observationType=full_stack \
  distill.bc_training=eval \
  distill.ablation_mode=prop-only \
  distill.student_resume=runs/student/bc-cross-prop-only/model_bc_14100.pth \
  distill.video_path=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_cross_20260614/prop_only_zoom.mp4
