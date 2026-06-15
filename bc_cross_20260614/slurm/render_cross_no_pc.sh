#!/bin/bash
#SBATCH --job-name=render-cross-no-pc
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-l40-0.grasp.maas
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_cross_20260614/logs/%j_render_no_pc.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_cross_20260614/logs/%j_render_no_pc.err

export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}

source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

echo "[$(date)] Render no-pc starting"

STATES_PATH=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_cross_20260614/no_pc_states.pt
VIDEO_PATH=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_cross_20260614/no_pc.mp4

COMMON="headless=True task=AllegroArmMOAR task.env.objSet=cross \
  task.env.legacy_obs=True task.env.observationType=full_stack \
  task.env.is_distillation=True task.env.numEnvs=1 \
  task.env.record_video=True \
  distill.bc_training=render_saved distill.ablation_mode=no-pc \
  distill.states_path=${STATES_PATH} distill.video_path=${VIDEO_PATH} \
  distill.teacher_data_dir=demonstration-cross-s1-multimodplus \
  distill.teacher_logdir=runs distill.teacher_resume=wheel-wrench-teacher-s1 \
  distill.student_logdir=runs/student/bc-cross-no-pc \
  distill.student_resume=runs/student/bc-cross-no-pc/model_bc_14050.pth \
  train.params.config.minibatch_size=1024 \
  train.params.config.central_value_config.minibatch_size=1024 \
  train.params.config.user_prefix=render-cross-no-pc \
  experiment=render-cross-no-pc wandb_activate=False"

CUDA_VISIBLE_DEVICES=0 python -u ./isaacgymenvs/train_distillation.py ${COMMON}

echo "[$(date)] Render no-pc done. Video at ${VIDEO_PATH}"
