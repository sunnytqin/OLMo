#!/bin/bash

#SBATCH -p kempner          # Use CPU partition
#SBATCH --nodes=1           # Single node
#SBATCH --ntasks=1          # Single task
#SBATCH --cpus-per-task=1   # Single CPU
#SBATCH --mem=2G            # Minimal memory
#SBATCH --time=10:00:00    # Run for 7 days (adjust as needed)
#SBATCH --gres=gpu:1  
#SBATCH --account=kempner_dam_lab
#SBATCH -o slurm_out/monitor-%j.out
#SBATCH -e slurm_out/monitor-%j.out
#SBATCH --job-name=olmo-monitor
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tqin@g.harvard.edu

# Configuration
TRAINING_SCRIPT="/n/home05/sqin/OLMo/experiment_scripts/pretrain-multinode.sh"
JOB_NAME="olmo-1B"  # Must match the job name in pretrain-multinode.sh
CHECK_INTERVAL=900  # 15 minutes in seconds
TRAINING_JOB_ID=""

echo "Monitor started at $(date)"
echo "Will check every $CHECK_INTERVAL seconds (15 minutes)"
echo "Training script: $TRAINING_SCRIPT"
echo "Job name to monitor: $JOB_NAME"
echo ""

# Function to check if job is running
is_job_running() {
    if [ -z "$TRAINING_JOB_ID" ]; then
        # No job ID yet, check by name
        squeue -u $USER -n "$JOB_NAME" -h | wc -l
    else
        # Check specific job ID
        squeue -j "$TRAINING_JOB_ID" -h 2>/dev/null | wc -l
    fi
}

# Function to submit training job
submit_training_job() {
    echo "[$(date)] Submitting training job..."
    JOB_OUTPUT=$(sbatch "$TRAINING_SCRIPT")
    if [ $? -eq 0 ]; then
        TRAINING_JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
        echo "[$(date)] Successfully submitted job ID: $TRAINING_JOB_ID"
    else
        echo "[$(date)] ERROR: Failed to submit job"
        TRAINING_JOB_ID=""
    fi
}

# Main monitoring loop
while true; do
    RUNNING_COUNT=$(is_job_running)

    if [ "$RUNNING_COUNT" -eq 0 ]; then
        echo "[$(date)] Training job not running. Resubmitting..."
        submit_training_job
    else
        if [ -n "$TRAINING_JOB_ID" ]; then
            echo "[$(date)] Training job $TRAINING_JOB_ID is running"
        else
            echo "[$(date)] Training job '$JOB_NAME' is running"
        fi
    fi

    echo "[$(date)] Sleeping for 30 minutes..."
    sleep $CHECK_INTERVAL
done
