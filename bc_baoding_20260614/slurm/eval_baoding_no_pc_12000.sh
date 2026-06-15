#!/bin/bash
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-l40-0.grasp.maas
#SBATCH --job-name=eval-baoding-no-pc-12000
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_baoding_20260614/logs/%j_eval-baoding-no-pc-12000.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_baoding_20260614/logs/%j_eval-baoding-no-pc-12000.err

export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}
source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

echo "[$(date)] eval-baoding-no-pc-12000 starting"

COMMON="headless=True task=AllegroArmMOAR task.env.objSet=ball \
  task.env.legacy_obs=True task.env.observationType=full_stack_baoding \
  task.env.is_distillation=True task.env.numEnvs=64 \
  distill.bc_training=eval distill.ablation_mode=no-pc \
  distill.teacher_data_dir=demonstration-baoding-s1-multimodplus \
  distill.teacher_logdir=runs distill.teacher_resume=baoding-teacher-s1 \
  distill.student_logdir=runs/student/bc-baoding-no-pc \
  distill.student_resume=runs/student/bc-baoding-no-pc/model_bc_12000.pth \
  train.params.config.minibatch_size=1024 \
  train.params.config.central_value_config.minibatch_size=1024 \
  train.params.config.user_prefix=eval-baoding-no-pc-12000 \
  experiment=eval-baoding-no-pc-12000 wandb_activate=False"

CUDA_VISIBLE_DEVICES=0 python -u ./isaacgymenvs/train_distillation.py ${COMMON}

echo "[$(date)] eval-baoding-no-pc-12000 done."
