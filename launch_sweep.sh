#!/bin/bash
# ============================================================================
# SAM3 Hyperparameter Sweep Launcher
# Submits multiple SLURM jobs via train_sweep.slurm, one per experiment.
#
# Usage:
#   bash launch_sweep.sh          # submit all experiments (03-09)
#   bash launch_sweep.sh 3 5 7    # submit only exp 03, 05, 07
#   bash launch_sweep.sh 03 05 07 # same as above (leading zeros accepted)
#   bash launch_sweep.sh --dry    # print commands without submitting
#
# Completed experiments (not in this script, results already exist):
#   01  v1_clean_noaug   NOAUG  100ep  ann=clean  -> scratch/sweep/baseline_100ep  (job 1523991)
#   02  v2_clean_aug     AUG    100ep  ann=clean  -> scratch/sweep/aug_100ep       (job 1523992)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_ROOT="${SCRIPT_DIR}/scratch/sweep"
SLURM_TEMPLATE="${SCRIPT_DIR}/train_sweep.slurm"
AUG_CONFIG="configs/roboflow_v100/firebox_finetuning"
NOAUG_CONFIG="configs/roboflow_v100/firebox_finetuning_worked_previous"

CLEAN_ANN="/home/yuqyan/Yuqi/LSB-AI-Detection/data/02_processed/sam3_prepared/annotations_train.json"
NOISE_ANN="/home/yuqyan/Yuqi/LSB-AI-Detection/data/02_processed/sam3_prepared/annotations_train_noise_augmented.json"

DRY_RUN=false
SELECTED_IDS=()
for arg in "$@"; do
    if [ "$arg" = "--dry" ]; then
        DRY_RUN=true
    else
        SELECTED_IDS+=("$arg")
    fi
done

# ============================================================================
# Experiment definitions
#   Format: ID|NAME|CONFIG|TIME|OVERRIDES
#   Every experiment explicitly pins trainer.data.train.dataset.ann_file
#   so results are reproducible regardless of annotations_train_active.json
#   symlink state.
#
# ---- COMPLETED (results already in scratch/sweep/) ----
# 01  v1_clean_noaug   NOAUG  100ep  ann=clean   (= baseline_100ep, job 1523991)
# 02  v2_clean_aug     AUG    100ep  ann=clean   (= aug_100ep,      job 1523992)
#
# ---- Core noise ablation ----
# 03  v3_noise_noaug   NOAUG  100ep  ann=noise   noise data, no color aug
# 04  v4_noise_aug     AUG    100ep  ann=noise   noise data + ColorJitter + RandomGrayscale
#
# ---- Hyperparameter tuning on v4 (noise+aug) base ----
# 05  v4_200ep         AUG    200ep  ann=noise   longer training (max_epochs: 100->200)
# 06  v4_dice80        AUG    100ep  ann=noise   boundary loss   (loss_dice: 50->80)
# 07  v4_boxnoise      AUG    100ep  ann=noise   box robustness  (box_noise_std: 0.05->0.1, box_noise_max: 10->20)
# 08  v4_higher_vlr    AUG    100ep  ann=noise   vision backbone (lr_scale: 0.1->0.3, vision LR 2.5e-7->7.5e-7)
# 09  v4_dice80_200ep  AUG    200ep  ann=noise   combined best   (loss_dice: 50->80, max_epochs: 100->200)
# ============================================================================
declare -a EXPERIMENTS=(

# --- 03: noise data, no color augmentation ---
"03|v3_noise_noaug|${NOAUG_CONFIG}|10:00:00|trainer.data.train.dataset.ann_file=${NOISE_ANN}"

# --- 04: noise data + ColorJitter + RandomGrayscale ---
"04|v4_noise_aug|${AUG_CONFIG}|10:00:00|trainer.data.train.dataset.ann_file=${NOISE_ANN}"

# --- 05: v4 + 200 epochs ---
"05|v4_200ep|${AUG_CONFIG}|20:00:00|trainer.data.train.dataset.ann_file=${NOISE_ANN} trainer.max_epochs=200"

# --- 06: v4 + loss_dice 50->80 ---
"06|v4_dice80|${AUG_CONFIG}|10:00:00|trainer.data.train.dataset.ann_file=${NOISE_ANN} roboflow_train.loss.loss_fns_find.2.weight_dict.loss_dice=80.0"

# --- 07: v4 + box_noise_std 0.05->0.1, box_noise_max 10->20 ---
"07|v4_boxnoise|${AUG_CONFIG}|10:00:00|trainer.data.train.dataset.ann_file=${NOISE_ANN} roboflow_train.train_transforms.0.transforms.1.box_noise_std=0.1 roboflow_train.train_transforms.0.transforms.1.box_noise_max=20"

# --- 08: v4 + lr_scale 0.1->0.3 ---
"08|v4_higher_vlr|${AUG_CONFIG}|10:00:00|trainer.data.train.dataset.ann_file=${NOISE_ANN} scratch.lr_scale=0.3"

# --- 09: v4 + loss_dice 50->80 + 200 epochs ---
"09|v4_dice80_200ep|${AUG_CONFIG}|20:00:00|trainer.data.train.dataset.ann_file=${NOISE_ANN} trainer.max_epochs=200 roboflow_train.loss.loss_fns_find.2.weight_dict.loss_dice=80.0"

)

# ============================================================================
# Submit loop
# ============================================================================
mkdir -p logs/sweep
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "SAM3 Sweep Launcher  (${TIMESTAMP})"
echo "Sweep root: ${SWEEP_ROOT}"
echo "============================================"

SUBMITTED=0
for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r EXP_ID EXP_NAME EXP_CONFIG EXP_TIME EXP_OVERRIDES <<< "$entry"

    # Filter by selected IDs if any provided (accept both "3" and "03")
    if [ ${#SELECTED_IDS[@]} -gt 0 ]; then
        MATCH=false
        for sel in "${SELECTED_IDS[@]}"; do
            NORM_SEL=$(echo "$sel" | sed 's/^0*//')
            NORM_ID=$(echo "$EXP_ID" | sed 's/^0*//')
            if [ "$NORM_SEL" = "$NORM_ID" ]; then MATCH=true; break; fi
        done
        if [ "$MATCH" = false ]; then continue; fi
    fi

    SWEEP_DIR="${SWEEP_ROOT}/${EXP_NAME}"
    JOB_NAME="sam3_${EXP_NAME}"

    echo ""
    echo "--- [${EXP_ID}] ${EXP_NAME} ---"
    echo "  Config:    ${EXP_CONFIG}"
    echo "  Time:      ${EXP_TIME}"
    echo "  Overrides: ${EXP_OVERRIDES:-<none>}"
    echo "  Output:    ${SWEEP_DIR}"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] would submit: sbatch --job-name=${JOB_NAME} ..."
    else
        JOB_ID=$(sbatch \
            --job-name="${JOB_NAME}" \
            --output="logs/sweep/${EXP_NAME}_%j.out" \
            --error="logs/sweep/${EXP_NAME}_%j.err" \
            --time="${EXP_TIME}" \
            --export=ALL,SWEEP_NAME="${EXP_NAME}",CONFIG="${EXP_CONFIG}",OVERRIDES="${EXP_OVERRIDES}",SWEEP_DIR="${SWEEP_DIR}" \
            "${SLURM_TEMPLATE}" | awk '{print $NF}')

        echo "  Submitted: SLURM Job ${JOB_ID}"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

echo ""
echo "============================================"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN complete. No jobs submitted."
else
    echo "Submitted ${SUBMITTED} jobs."
    echo "Monitor: squeue -u \$USER"
    echo "TensorBoard: tensorboard --logdir ${SWEEP_ROOT} --port 6006"
    echo "Manifest: cat logs/sweep/manifest.txt"
fi
echo "============================================"
