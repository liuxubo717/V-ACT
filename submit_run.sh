#!/bin/bash
#SBATCH --job-name=ACT
#SBATCH --partition=lowpri
#SBATCH --account=all
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --time=23:59:59
#SBATCH --exclude=a100-st-p4d24xlarge-204,a100-st-p4d24xlarge-58,a100-st-p4d24xlarge-77,a100-st-p4d24xlarge-74,a100-st-p4d24xlarge-233,a100-st-p4d24xlarge-143,a100-st-p4d24xlarge-103,a100-st-p4d24xlarge-138,a100-st-p4d24xlarge-278
#SBATCH --error=/data/home/xuboliu/slurm/err/%j_%t_log.err
#SBATCH --output=/data/home/xuboliu/slurm/out/%j_%t_log.out
export HYDRA_FULL_ERROR=1
source /data/home/xuboliu/miniconda/etc/profile.d/conda.sh;conda activate ACT
cd /data/home/xuboliu/project/V-ACT/
srun python train.py -m audio_visual -v S3D_25frames -n ACT-S3D_25frames
srun python train.py -m audio_visual -v S3D_10frames -n ACT-S3D_10frames
srun python train.py -m video -v S3D_25frames -n S3D_25frames
srun python train.py -m video -v S3D_10frames -n S3D_10frames









