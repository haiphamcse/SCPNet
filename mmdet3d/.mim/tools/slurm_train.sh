#!/bin/bash -e
#SBATCH --job-name=SceneOcc
#SBATCH --output=/lustre/scratch/client/vinai/users/tuannd42/SceneOcc/slurm_out/slurm_%A.out
#SBATCH --error=/lustre/scratch/client/vinai/users/tuannd42/SceneOcc/slurm_out/slurm_%A.err

#SBATCH --gpus=4
#SBATCH --nodes=1

#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-gpu=32

#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.tuannd42@vinai.io

srun --container-image=/lustre/scratch/client/vinai/users/tuannd42/docker_images/scene_occ_new2.sqsh \
--container-mounts=/lustre/scratch/client/vinai/users/tuannd42/SceneOcc:/home/ubuntu/SceneOcc \
--container-workdir=/home/ubuntu/SceneOcc/BEVDet \
./tools/dist_train.sh configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_scale_dice_ce_sem2d_densedepth_ft.py 4
