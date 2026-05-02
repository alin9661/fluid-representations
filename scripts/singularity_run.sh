#!/bin/bash
# Run a command inside the project's Singularity container with conda activated.
#
# Usage:
#   scripts/singularity_run.sh <command> [args...]
#
# Examples:
#   scripts/singularity_run.sh python train.py
#   scripts/singularity_run.sh python -m physics_ssl.data --compute-stats --split train
#
# Override paths via environment variables if your files live elsewhere:
#   SIF        — path to the base .sif image
#   EXT3       — path to the writable overlay (we mount :ro for batch jobs)
#   CONDA_ENV  — conda env name to activate after sourcing CONDA_INIT
#   CONDA_INIT — script that registers the `conda` shell function. Defaults to
#                /ext3/env.sh (the conventional NYU overlay layout). Override
#                with /share/apps/pyenv/py3.9/etc/profile.d/conda.sh when the
#                conda install lives on the host instead of inside the overlay.

set -euo pipefail

SIF=${SIF:-$SCRATCH/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}
EXT3=${EXT3:-$SCRATCH/deep-learning/physical-learning-15GB-500K.ext3}
CONDA_ENV=${CONDA_ENV:-physical-representation-learning}
CONDA_INIT=${CONDA_INIT:-/ext3/env.sh}

# :ro is important for batch jobs — it lets multiple jobs share the same
# overlay concurrently. Drop :ro when you need to install packages interactively.
OVERLAY_MODE=${OVERLAY_MODE:-ro}

if [[ ! -f "$SIF" ]]; then
  echo "SIF not found: $SIF" >&2
  exit 1
fi
if [[ ! -f "$EXT3" ]]; then
  echo "EXT3 not found: $EXT3" >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 1
fi

singularity exec --nv \
  --overlay "${EXT3}:${OVERLAY_MODE}" \
  "$SIF" \
  /bin/bash -c '
        set -euo pipefail
        source '"$CONDA_INIT"'
        conda activate '"$CONDA_ENV"'
        exec "$@"
    ' _ "$@"
