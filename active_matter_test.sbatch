#!/bin/bash
#SBATCH --job-name=tjepa-active-test
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=./%j_%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aaron.lin@nyu.edu

# Usage:
#   sbatch active_matter_test.sbatch
#   sbatch active_matter_test.sbatch test.checkpoint_path=checkpoints/<run>/best/<file>.ckpt
#
# Auto-discovers the freshest .ckpt under ${cfg.checkpoint.dirpath}/best/ unless you
# pass test.checkpoint_path=... explicitly. Note that cfg.checkpoint.dirpath uses a
# ${now:...} timestamp by default, so for a real test run you almost always want to
# pin checkpoint.dirpath=<the training run's dir> alongside test.checkpoint_path or
# pass test.checkpoint_path directly.

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Same singularity/conda setup as active_matter_tjepa.sbatch.
export SIF=${SIF:-$SCRATCH/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}
export EXT3=${EXT3:-$SCRATCH/deep-learning/physical-learning-15GB-500K.ext3}
export CONDA_ENV=${CONDA_ENV:-physical-representation-learning}
export THE_WELL_DATA_DIR=${THE_WELL_DATA_DIR:-/scratch/$USER/deep-learning/the_well_data/datasets}
export STATS_PATH=${STATS_PATH:-cache/active_matter_stats.npz}

HYDRA_OVERRIDES=("$@")

echo "=== T-JEPA Active Matter TEST ==="
echo "Project:    $(pwd)"
echo "SIF:        ${SIF}"
echo "EXT3:       ${EXT3}"
echo "CONDA_ENV:  ${CONDA_ENV}"
echo "WELL_DATA:  ${THE_WELL_DATA_DIR}"
echo "Stats:      ${STATS_PATH}"
echo "Overrides:  ${HYDRA_OVERRIDES[*]:-<none>}"
echo "Host:       $(hostname)"
echo "GPUs:       ${SLURM_GPUS_PER_NODE:-?}"

# Fail fast on missing prerequisites — silent skips would mask config errors.
[[ -f "${SIF}"  ]] || { echo "ERROR: SIF not found: ${SIF}"   >&2; exit 1; }
[[ -f "${EXT3}" ]] || { echo "ERROR: EXT3 not found: ${EXT3}" >&2; exit 1; }
[[ -d "${THE_WELL_DATA_DIR}/active_matter/data/test" ]] || {
    echo "ERROR: test split not found at ${THE_WELL_DATA_DIR}/active_matter/data/test" >&2
    exit 1
}
[[ -f "${STATS_PATH}" ]] || {
    echo "ERROR: stats file ${STATS_PATH} missing — run training (or stats bootstrap) first" >&2
    exit 1
}
[[ -x scripts/singularity_run.sh ]] || {
    echo "ERROR: scripts/singularity_run.sh not found or not executable in $(pwd)" >&2
    echo "       did you submit from the project root?" >&2
    exit 1
}

mkdir -p logs

# Run evaluation inside the container.
scripts/singularity_run.sh \
    python test.py \
        data.stats_path="${STATS_PATH}" \
        "${HYDRA_OVERRIDES[@]}"
