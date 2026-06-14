#!/bin/bash
#SBATCH --job-name=collect-baoding-w2
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-l40-0.grasp.maas
#SBATCH --output=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_baoding_20260614/logs/%j_collect_w2.out
#SBATCH --error=/mnt/kostas-graid/sw/envs/edward/in-hand-rotation-results/bc_baoding_20260614/logs/%j_collect_w2.err

export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}

source /mnt/kostas-graid/sw/envs/edward/in-hand-rotation/envs_robosyn/bin/activate
cd /mnt/kostas-graid/sw/envs/edward/in-hand-rotation

echo "[$(date)] Worker 2 starting"

COMMON="headless=True task=AllegroArmMOAR task.env.objSet=ball task.env.legacy_obs=False task.env.observationType=full_stack_baoding task.env.is_distillation=True task.env.ablation_mode=multi-modality-plus distill.ablation_mode=multi-modality-plus distill.bc_training=collect distill.teacher_data_dir=demonstration-baoding-s1-multimodplus distill.teacher_logdir=runs distill.teacher_resume=baoding-teacher-s1 task.env.numEnvs=64 train.params.config.user_prefix=bc-baoding-collect train.params.config.minibatch_size=1024 train.params.config.central_value_config.minibatch_size=1024 experiment=bc-baoding-collect wandb_activate=False distill.learn.max_iterations=500"

CUDA_VISIBLE_DEVICES=0 python -u ./isaacgymenvs/train_distillation.py ${COMMON} distill.worker_id=2

echo "[$(date)] Worker 2 done."
