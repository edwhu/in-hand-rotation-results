#!/bin/bash
#SBATCH --job-name=teacher-cross
#SBATCH --partition=dgx-b200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/vast/projects/dineshj/lab/hued/in-hand-rotation-results/teacher_cross_20260613/logs/%j.out
#SBATCH --error=/vast/projects/dineshj/lab/hued/in-hand-rotation-results/teacher_cross_20260613/logs/%j.err

source /vast/home/h/hued/miniconda3/etc/profile.d/conda.sh
conda activate /vast/projects/dineshj/lab/hued/env_robosyn
export LD_LIBRARY_PATH=/vast/projects/dineshj/lab/hued/env_robosyn/lib:$LD_LIBRARY_PATH

cd /vast/projects/dineshj/lab/hued/in-hand-rotation

CUDA_VISIBLE_DEVICES=0 python ./isaacgymenvs/train.py headless=True \
  task.env.objSet=cross task=AllegroArmMOAR task.env.axis=z \
  task.env.numEnvs=8192 train.params.config.minibatch_size=16384 \
  train.params.config.central_value_config.minibatch_size=16384 \
  task.env.observationType=full_stack task.env.legacy_obs=True \
  task.env.ablation_mode=no-pc experiment=wheel-wrench \
  train.params.config.user_prefix=wheel-wrench wandb_activate=True \
  wandb_entity=edhu

echo "[$(date)] Done."
