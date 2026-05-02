#!/bin/bash
#SBATCH --job-name=tjepa-active
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=./%j_%x.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aaron.lin@nyu.edu

# Usage:
#   sbatch active_matter_tjepa.sbatch
#   sbatch active_matter_tjepa.sbatch trainer.max_epochs=50 loader.batch_size=8
#
# Project-rule-compliant overrides (see RegressionKNN discussion):
#   sbatch active_matter_tjepa.sbatch \
#       predictor.unconditional=true \
#       probes.linear.enabled=false probes.knn.enabled=false

# Slurm copies this sbatch into its spool dir before running, so anchor cwd
# back to where the user invoked sbatch. SLURM_SUBMIT_DIR is set by Slurm and
# falls back to the current $PWD for interactive testing.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Override these if your files live somewhere other than $SCRATCH:
export SIF=${SIF:-$SCRATCH/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}
export EXT3=${EXT3:-$SCRATCH/deep-learning/physical-learning-15GB-500K.ext3}
export CONDA_ENV=${CONDA_ENV:-physical-representation-learning}
export THE_WELL_DATA_DIR=${THE_WELL_DATA_DIR:-/scratch/$USER/deep-learning/the_well_data/datasets}

# Per-channel z-score stats are cached here, relative to project root.
# Auto-bootstrapped below if missing (one-time, ~1-2 min).
export STATS_PATH=${STATS_PATH:-cache/active_matter_stats.npz}

# Forward extra positional args to Hydra (e.g. trainer.max_epochs=50).
HYDRA_OVERRIDES=("$@")

echo "=== T-JEPA Active Matter pretraining ==="
echo "Project:    $(pwd)"
echo "SIF:        ${SIF}"
echo "EXT3:       ${EXT3}"
echo "CONDA_ENV:  ${CONDA_ENV}"
echo "WELL_DATA:  ${THE_WELL_DATA_DIR}"
echo "Stats:      ${STATS_PATH}"
echo "Overrides:  ${HYDRA_OVERRIDES[*]:-<none>}"
echo "Host:       $(hostname)"
echo "GPUs:       ${SLURM_GPUS_PER_NODE:-?}"

# Fail fast if required paths are missing
[[ -f "${SIF}"  ]] || { echo "ERROR: SIF not found: ${SIF}"   >&2; exit 1; }
[[ -f "${EXT3}" ]] || { echo "ERROR: EXT3 not found: ${EXT3}" >&2; exit 1; }
[[ -d "${THE_WELL_DATA_DIR}/active_matter/data" ]] || {
    echo "ERROR: active_matter data dir not found at ${THE_WELL_DATA_DIR}/active_matter/data" >&2
    exit 1
}
[[ -x scripts/singularity_run.sh ]] || {
    echo "ERROR: scripts/singularity_run.sh not found or not executable in $(pwd)" >&2
    echo "       did you submit from the project root?" >&2
    exit 1
}

mkdir -p logs cache checkpoints

# Auto-bootstrap channel stats if they don't exist yet.
if [[ ! -f "${STATS_PATH}" ]]; then
    echo "Stats file not found; computing per-channel z-score statistics..."
    scripts/singularity_run.sh \
        python -m physics_ssl.data \
            --compute-stats --split train \
            --root "${THE_WELL_DATA_DIR}/active_matter/data" \
            --out "${STATS_PATH}"
fi

# Run training inside the container (singularity_run.sh handles conda activation
# via /ext3/env.sh — same wrapper as active_matter_finetune_jepa.sbatch).
scripts/singularity_run.sh \
    python train.py \
        data.stats_path="${STATS_PATH}" \
        "${HYDRA_OVERRIDES[@]}"
