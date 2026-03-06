#!/bin/bash
#SBATCH --job-name=sam3_fast
#SBATCH --output=logs/sam3_finetune_%j.out
#SBATCH --error=logs/sam3_finetune_%j.err
#SBATCH --time=10:00:00        
#SBATCH --gpus=H100:1               
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G

#SBATCH --partition=lowprio    
# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuqi.yang@uzh.ch

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "============================================"

# Create logs directory if not exists
mkdir -p logs

# Navigate to project directory
cd /home/yuqyan/Yuqi/sam3/sam3

module load miniforge3

eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate sam3

# Print GPU info
echo "GPU Info:"
nvidia-smi

# Run training
python train/train.py -c configs/roboflow_v100/firebox_finetuning --use-cluster 0 --num-gpus 1

# Print completion info
echo "============================================"
echo "End Time: $(date)"
echo "Training completed!"
echo "============================================"
