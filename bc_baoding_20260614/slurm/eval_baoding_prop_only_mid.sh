#!/bin/bash
#SBATCH --job-name=eval-baoding-prop-only-mid
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-l40-0.grasp.maas
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_baoding_20260614/logs/%j_eval_baoding_prop_only_mid.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_baoding_20260614/logs/%j_eval_baoding_prop_only_mid.err

export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}

source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

echo "[$(date)] Eval baoding prop-only mid starting"

COMMON="headless=True task=AllegroArmMOAR task.env.objSet=ball \
  task.env.legacy_obs=True task.env.observationType=full_stack_baoding \
  task.env.is_distillation=True task.env.numEnvs=64 \
  distill.bc_training=eval distill.ablation_mode=prop-only \
  distill.teacher_data_dir=demonstration-baoding-s1-multimodplus \
  distill.teacher_logdir=runs distill.teacher_resume=baoding-teacher-s1 \
  distill.student_logdir=runs/student/bc-baoding-prop-only \
  distill.student_resume=runs/student/bc-baoding-prop-only/model_bc_3250.pth \
  train.params.config.minibatch_size=1024 \
  train.params.config.central_value_config.minibatch_size=1024 \
  train.params.config.user_prefix=eval-baoding-prop-only-mid \
  experiment=eval-baoding-prop-only-mid wandb_activate=False"

CUDA_VISIBLE_DEVICES=0 python -u ./isaacgymenvs/train_distillation.py ${COMMON}

echo "[$(date)] Eval baoding prop-only mid done."
