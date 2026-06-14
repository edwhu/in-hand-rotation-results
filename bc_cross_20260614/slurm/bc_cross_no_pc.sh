#!/bin/bash
#SBATCH --job-name=bc-cross-no-pc
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-l40-0.grasp.maas
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_cross_20260614/logs/%j_bc_no_pc.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_cross_20260614/logs/%j_bc_no_pc.err

export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}

source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

echo "[$(date)] BC no-pc starting"

COMMON="headless=True task=AllegroArmMOAR task.env.objSet=cross \
  task.env.legacy_obs=False task.env.observationType=full_stack \
  task.env.is_distillation=True task.env.numEnvs=64 \
  distill.bc_training=warmup distill.ablation_mode=no-pc \
  distill.teacher_data_dir=demonstration-cross-s1-multimodplus \
  distill.teacher_logdir=runs distill.teacher_resume=wheel-wrench-teacher-s1 \
  distill.student_logdir=runs/student/bc-cross-no-pc \
  train.params.config.minibatch_size=1024 \
  train.params.config.central_value_config.minibatch_size=1024 \
  train.params.config.user_prefix=bc-cross-no-pc \
  experiment=bc-cross-no-pc wandb_activate=False"

CUDA_VISIBLE_DEVICES=0 python -u ./isaacgymenvs/train_distillation.py ${COMMON}

echo "[$(date)] BC no-pc done."
