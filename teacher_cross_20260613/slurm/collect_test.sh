#!/bin/bash
#SBATCH --job-name=collect-test
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-l40-0.grasp.maas
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/teacher_cross_20260613/logs/%j_collect_test.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/teacher_cross_20260613/logs/%j_collect_test.err

# Use per-job local tmp dir to avoid NFS lock contention on gymtorch compilation
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}

source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

echo "[$(date)] Starting test collection (max_iterations=2)"

CUDA_VISIBLE_DEVICES=0 python ./isaacgymenvs/train_distillation.py headless=True \
  distill.teacher_data_dir=demonstration-cross-test \
  distill.teacher_logdir=runs \
  distill.teacher_resume=wheel-wrench-teacher-s1 \
  task.env.legacy_obs=False distill.bc_training=collect \
  task.env.objSet=cross task.env.is_distillation=True \
  task=AllegroArmMOAR task.env.numEnvs=64 \
  task.env.observationType=full_stack \
  distill.worker_id=0 distill.ablation_mode=multi-modality-plus \
  task.env.ablation_mode=multi-modality-plus \
  train.params.config.user_prefix=collect-test \
  train.params.config.minibatch_size=1024 \
  train.params.config.central_value_config.minibatch_size=1024 \
  experiment=collect-test wandb_activate=False \
  distill.learn.max_iterations=2

echo "[$(date)] Done."
