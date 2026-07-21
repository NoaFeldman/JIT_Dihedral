#!/bin/bash
#SBATCH --job-name=mljit
#SBATCH --array=1-200
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/mljit_%A_%a.out

set -euo pipefail

# --- Sweep configuration (editable) -----------------------------------------
# The array maps onto len(L_LIST) * len(P_LIST) configurations; NUM_TASKS must
# be divisible by that product (here 4 * 5 = 20 configurations, 10 chunks each,
# so each task runs REPS_PER_CONFIG / 10 repetitions).
L_LIST="5,7,9,11"
P_LIST="0.01,0.02,0.03,0.04,0.05"   # entry = single float, or per-layer probs like 0.02:0.01:0.01
NUM_LAYERS=3
REPS_PER_CONFIG=100000
NUM_TASKS=200
BOUNDARY="OBC"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/multilayer"
# ----------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}" "${SLURM_SUBMIT_DIR}/logs"
cd "${SLURM_SUBMIT_DIR}"

python cluster/multilayer_worker.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-tasks "${NUM_TASKS}" \
    --output-dir "${OUTPUT_DIR}" \
    --boundary "${BOUNDARY}" \
    --L-list "${L_LIST}" \
    --p-list "${P_LIST}" \
    --num-layers "${NUM_LAYERS}" \
    --reps-per-config "${REPS_PER_CONFIG}"
