#!/bin/bash

#SBATCH -c 64                          # Number of CPU cores
#SBATCH -t 3-0:00                     # Runtime (48 hours for 7B tokens)
#SBATCH -p kempner_h100                # Partition to submit to
#SBATCH --mem=250G                      # Memory
#SBATCH -n 1                            # Number of nodes
#SBATCH --gres=gpu:4                   # Number of GPUs (increase if needed)
#SBATCH -o ../slurm_out/midtrain-%j.out # Standard out
#SBATCH -e ../slurm_out/midtrain-%j.out # Standard err
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=olmo-midtrain

module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf


# Print job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS: $SLURM_GPUS"
echo ""

cd ../

export CACHED_PATH_CACHE_ROOT=/n/netscratch/dam_lab/Lab/sqin/olmo/
torchrun --nproc_per_node=4 /n/home05/sqin/OLMo/scripts/train.py experiment_scripts/OLMo2-7B-stage2-400k.yaml
