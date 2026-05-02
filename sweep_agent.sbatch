#!/bin/bash
#SBATCH --job-name=tjepa-sweep
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aaron.lin@nyu.edu

# Usage:
#   1. Create the sweep:
#        wandb sweep configs/sweep_tjepa_active_matter.yaml
#      → prints "Created sweep with ID: <entity>/<project>/<id>"
#
#   2. Submit the agent array (10 jobs, max 4 concurrent — adjust --array above):
#        SWEEP_ID=<entity>/<project>/<id> sbatch sweep_agent.sbatch
#
#   3. Each job runs a wandb agent that pulls trials from the sweep until it
#      hits TRIALS_PER_JOB or the 24h walltime, whichever comes first.
#
# Optional overrides:
#   TRIALS_PER_JOB=4 SWEEP_ID=... sbatch sweep_agent.sbatch
#
# This script intentionally mirrors active_matter_tjepa.sbatch's container,
# conda, data, and stats-bootstrap setup so sweep runs are bit-identical to
# normal training runs except for the launch command.

# Fail loudly on any error so a broken stats bootstrap or missing dependency
# can't silently launch wandb agents that then waste GPU hours on noise.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Sweep ID is required — fail fast with a useful message if it's missing.
if [[ -z "${SWEEP_ID:-}" ]]; then
    echo "ERROR: SWEEP_ID not set." >&2
    echo "       Run 'wandb sweep configs/sweep_tjepa_active_matter.yaml' first," >&2
    echo "       then submit with: SWEEP_ID=<entity/project/id> sbatch $0" >&2
    exit 1
fi

# Cap trials per agent so a single bad run can't burn the whole 24h walltime
# on one trial. The sweep server hands out new trials until exhausted.
TRIALS_PER_JOB=${TRIALS_PER_JOB:-8}

# Same container/conda/data layout as active_matter_tjepa.sbatch — keeps sweep
# runs reproducible against the regular training pipeline.
export SIF=${SIF:-$SCRATCH/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}
export EXT3=${EXT3:-$SCRATCH/deep-learning/physical-learning-15GB-500K.ext3}
export CONDA_ENV=${CONDA_ENV:-physical-representation-learning}
export THE_WELL_DATA_DIR=${THE_WELL_DATA_DIR:-/scratch/$USER/deep-learning/the_well_data/datasets}
export STATS_PATH=${STATS_PATH:-cache/active_matter_stats.npz}

echo "=== T-JEPA Active Matter W&B sweep agent ==="
echo "Project:        $(pwd)"
echo "SLURM job:      ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-N/A})"
echo "SWEEP_ID:       ${SWEEP_ID}"
echo "TRIALS_PER_JOB: ${TRIALS_PER_JOB}"
echo "SIF:            ${SIF}"
echo "EXT3:           ${EXT3}"
echo "CONDA_ENV:      ${CONDA_ENV}"
echo "WELL_DATA:      ${THE_WELL_DATA_DIR}"
echo "Stats:          ${STATS_PATH}"
echo "Host:           $(hostname)"
echo "GPUs:           ${SLURM_GPUS_PER_NODE:-?}"

[[ -f "${SIF}"  ]] || { echo "ERROR: SIF not found: ${SIF}"   >&2; exit 1; }
[[ -f "${EXT3}" ]] || { echo "ERROR: EXT3 not found: ${EXT3}" >&2; exit 1; }
[[ -d "${THE_WELL_DATA_DIR}/active_matter/data" ]] || {
    echo "ERROR: active_matter data dir not found at ${THE_WELL_DATA_DIR}/active_matter/data" >&2
    exit 1
}
[[ -x scripts/singularity_run.sh ]] || {
    echo "ERROR: scripts/singularity_run.sh not found or not executable in $(pwd)" >&2
    exit 1
}

mkdir -p logs cache checkpoints

# Auto-bootstrap channel z-score stats if missing — same one-shot step as the
# main training script. Sweep agents reuse the cached file across trials.
if [[ ! -f "${STATS_PATH}" ]]; then
    echo "Stats file not found; computing per-channel z-score statistics..."
    scripts/singularity_run.sh \
        python -m physics_ssl.data \
            --compute-stats --split train \
            --root "${THE_WELL_DATA_DIR}/active_matter/data" \
            --out "${STATS_PATH}"
    # Guard: if the bootstrap "succeeded" but didn't produce the file, abort
    # before launching trials. Otherwise train.py falls back to no z-score
    # (train.py warn at lines ~41-44) and every sweep trial logs noise.
    [[ -f "${STATS_PATH}" ]] || {
        echo "ERROR: stats bootstrap completed without producing ${STATS_PATH}" >&2
        exit 1
    }
fi

# Run the wandb agent inside the container. The agent polls the sweep server
# for hyperparameter assignments, launches `python train.py key=value ...`
# (per the `command:` block in the sweep YAML), and reports metrics back.
#
# WANDB_API_KEY must be available inside the container. The user's ~/.netrc
# (populated by `wandb login`) is bind-mounted by singularity_run.sh so the
# agent can authenticate without exposing the key in this script.
scripts/singularity_run.sh \
    wandb agent --count "${TRIALS_PER_JOB}" "${SWEEP_ID}"
