#!/bin/bash

#SBATCH -p kempner_requeue # Partition to submit to
#SBATCH --nodes=8                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=48        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=500G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --account=kempner_barak_lab
#SBATCH -o slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e slurm_out/slurm-%j.out # Standard err goes to this file
#SBATCH --job-name=olmo-1B    # create a short name for your job
#SBATCH --constraint=a100

# Enable command tracing for debugging
set -x

module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf

# Print job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_PROCID: $SLURM_PROCID"
echo ""

cd ../

export CACHED_PATH_CACHE_ROOT=/n/netscratch/dam_lab/Lab/sqin/olmo/

# Get the master node address (same approach as Ray script)
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
MASTER_NODE=${nodes_array[0]}
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)
MASTER_PORT=12345

echo "Master node: $MASTER_NODE"
echo "Master IP: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo ""

# Debug: Check node information
echo "=== Debugging node information ==="
srun bash -c 'echo "Node: $(hostname), SLURM_PROCID: $SLURM_PROCID, SLURM_NODEID: $SLURM_NODEID, SLURM_LOCALID: $SLURM_LOCALID"'
echo ""

echo "=== Network interfaces available ==="
srun bash -c 'echo "Node: $(hostname)"; ip addr show | grep -E "^[0-9]+: " | awk "{print \$2}" | sed "s/:$//"'
echo ""

echo "=== Network IPs (excluding loopback) ==="
srun bash -c 'echo "Node: $(hostname)"; ip addr show | grep "inet " | grep -v "127.0.0.1" | awk "{print \$NF, \$2}"'
echo ""

# Enable distributed debugging (set to INFO only when debugging issues)
# TORCH_DISTRIBUTED_DEBUG=INFO  # Uncomment for verbose FSDP/DDP debugging
# export NCCL_DEBUG=INFO         # Uncomment for verbose NCCL debugging
export NCCL_DEBUG=WARN           # Only show NCCL warnings/errors

# Network configuration for InfiniBand
export NCCL_SOCKET_IFNAME=^docker0,lo  # Exclude loopback and docker
export NCCL_IB_DISABLE=0               # Enable InfiniBand (0 = enabled)
export NCCL_IB_HCA=mlx5                # InfiniBand adapter (usually mlx5 for modern cards)
export NCCL_IB_GID_INDEX=3             # RoCE mode (3 is common for RoCEv2)
# If you want to explicitly use InfiniBand interfaces, uncomment:
# export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3

echo "=== Launching distributed training ==="
# Run distributed training
# Note: Using SLURM_PROCID instead of SLURM_NODEID for node_rank
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /n/home05/sqin/OLMo/scripts/train.py /n/home05/sqin/OLMo/experiment_scripts/OLMo2-1B-stage1-50B.yaml

# Capture exit code
EXIT_CODE=$?

echo ""
echo "Job finished at $(date)"
echo "Exit code: $EXIT_CODE"

# Auto-resubmit on failure (with retry limit)
if [ $EXIT_CODE -ne 0 ]; then
    # Get current retry count (default to 0 if not set)
    RETRY_COUNT=${SLURM_RESTART_COUNT:-0}
    MAX_RETRIES=5  # Maximum number of auto-retries (increased from 3 to 5)

    echo "Job failed with exit code $EXIT_CODE"
    echo "Current retry count: $RETRY_COUNT"

    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        NEXT_RETRY=$((RETRY_COUNT + 1))
        echo "Resubmitting job (attempt $NEXT_RETRY/$MAX_RETRIES)..."

        # Resubmit with incremented retry counter
        sbatch --export=ALL,SLURM_RESTART_COUNT=$NEXT_RETRY $0
        echo "Resubmitted as a new job"
    else
        echo "Maximum retries ($MAX_RETRIES) reached. Not resubmitting."
        echo "Please investigate the issue before manually resubmitting."
    fi
else
    echo "Job completed successfully!"
fi
