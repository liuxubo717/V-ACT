#!/bin/bash
#SBATCH --job-name=ACT
#SBATCH --partition=hipri
#SBATCH --account=all
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --time=00:00:00
#SBATCH --exclude=a100-st-p4d24xlarge-204,a100-st-p4d24xlarge-58,a100-st-p4d24xlarge-77,a100-st-p4d24xlarge-74,a100-st-p4d24xlarge-233,a100-st-p4d24xlarge-143,a100-st-p4d24xlarge-103,a100-st-p4d24xlarge-138,a100-st-p4d24xlarge-278
#SBATCH --error=/data/home/xuboliu/slurm/err/%j_%t_log.err
#SBATCH --output=/data/home/xuboliu/slurm/out/%j_%t_log.out
export HYDRA_FULL_ERROR=1
source /data/home/xuboliu/miniconda/etc/profile.d/conda.sh;conda activate ACT
cd /data/home/xuboliu/project/ACT/
srun python train.py -m audio_visual -n ACT-S3D -v S3D
# srun python train.py -m audio_visual -n ACT-I3D -v I3D



