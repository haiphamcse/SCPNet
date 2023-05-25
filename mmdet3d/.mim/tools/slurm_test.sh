#!/bin/bash -e
#SBATCH --job-name=SceneOcc
#SBATCH --output=/lustre/scratch/client/vinai/users/tuannd42/SceneOcc/slurm_out/slurm_%A.out
#SBATCH --error=/lustre/scratch/client/vinai/users/tuannd42/SceneOcc/slurm_out/slurm_%A.err

#SBATCH --gpus=5
#SBATCH --nodes=1

#SBATCH --mem-per-gpu=40G
#SBATCH --nodelist=sdc2-hpc-dgx-a100-008
#SBATCH --cpus-per-gpu=32

#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.tuannd42@vinai.io

srun --container-image=/lustre/scratch/client/vinai/users/tuannd42/docker_images/scene_occ_new2.sqsh \
--container-mounts=/lustre/scratch/client/vinai/users/tuannd42/SceneOcc:/home/ubuntu/SceneOcc \
--container-workdir=/home/ubuntu/SceneOcc/BEVDet \
./tools/dist_test.sh configs/bevdet_occ/bevdet-occ-stbase-4d-stereo-512x1408-24e_scale_dice_ce_longterm_freeze_bb.py work_dirs/bevdet-occ-stbase-4d-stereo-512x1408-24e_scale_dice_ce_longterm_freeze_bb/epoch_24_ema.pth 5 --eval box
