#!/bin/bash
#SBATCH --gpus=H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=lowprio
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuqi.yang@uzh.ch

# ---- job-name, output, error, time are set by launch_sweep.sh via sbatch CLI ----
# Required env vars from launcher: SWEEP_NAME, CONFIG, OVERRIDES, SWEEP_DIR

echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Job Name:    $SLURM_JOB_NAME"
echo "Experiment:  $SWEEP_NAME"
echo "Config:      $CONFIG"
echo "Overrides:   $OVERRIDES"
echo "Sweep Dir:   $SWEEP_DIR"
echo "Node:        $SLURMD_NODENAME"
echo "Start Time:  $(date)"
echo "============================================"

mkdir -p logs/sweep

# Append to manifest: experiment name -> SLURM job ID mapping
echo "${SWEEP_NAME}  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME}  started=$(date -Iseconds)" >> logs/sweep/manifest.txt

cd /home/yuqyan/Yuqi/sam3/sam3

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate sam3

echo "GPU Info:"
nvidia-smi

# Build override list: always set experiment_log_dir; append user overrides
OVERRIDE_ARGS="paths.experiment_log_dir=${SWEEP_DIR}"
if [ -n "$OVERRIDES" ]; then
    OVERRIDE_ARGS="${OVERRIDE_ARGS} ${OVERRIDES}"
fi

python train/train.py \
    -c "$CONFIG" \
    --use-cluster 0 \
    --num-gpus 1 \
    --overrides $OVERRIDE_ARGS

EXIT_CODE=$?

echo "============================================"
echo "End Time:    $(date)"
echo "Exit Code:   $EXIT_CODE"
echo "Experiment:  $SWEEP_NAME"
echo "============================================"

exit $EXIT_CODE
