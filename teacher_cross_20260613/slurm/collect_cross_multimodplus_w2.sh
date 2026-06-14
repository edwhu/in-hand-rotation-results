#!/bin/bash
#SBATCH --job-name=collect-cross-wW
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-l40-0.grasp.maas
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/teacher_cross_20260613/logs/%j_collect_wW.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/teacher_cross_20260613/logs/%j_collect_wW.err

source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

CUDA_VISIBLE_DEVICES=0 python ./isaacgymenvs/train_distillation.py headless=True \
  distill.teacher_data_dir=demonstration-cross-s1-multimodplus \
  distill.teacher_logdir=runs \
  distill.teacher_resume=wheel-wrench-teacher-s1 \
  task.env.legacy_obs=False distill.bc_training=collect \
  task.env.objSet=cross task.env.is_distillation=True \
  task=AllegroArmMOAR task.env.numEnvs=64 \
  task.env.observationType=full_stack \
  distill.worker_id=2 distill.ablation_mode=multi-modality-plus \
  task.env.ablation_mode=multi-modality-plus \
  train.params.config.user_prefix=bc-cross-collect \
  train.params.config.minibatch_size=1024 \
  train.params.config.central_value_config.minibatch_size=1024 \
  experiment=bc-cross-collect wandb_activate=False \
  distill.learn.max_iterations=100

echo "[$(date)] Worker 2 done."
