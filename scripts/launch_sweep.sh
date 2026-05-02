#!/usr/bin/env bash
# Two-step convenience wrapper around `wandb sweep` + `sbatch sweep_agent.sbatch`.
#
# Step 1: creates the sweep on the W&B server and parses the printed sweep ID.
# Step 2: prints (does NOT run) the sbatch command the user should submit to
#         launch the agent array. Submission is left manual so the user can
#         tweak --array bounds or TRIALS_PER_JOB before committing GPU hours.
#
# Usage:
#   scripts/launch_sweep.sh                                  # default sweep config
#   scripts/launch_sweep.sh configs/my_other_sweep.yaml      # alternative config

set -euo pipefail

SWEEP_CFG="${1:-configs/sweep_tjepa_active_matter.yaml}"

if [[ ! -f "${SWEEP_CFG}" ]]; then
    echo "ERROR: sweep config not found: ${SWEEP_CFG}" >&2
    exit 1
fi

if ! command -v wandb >/dev/null 2>&1; then
    echo "ERROR: 'wandb' CLI not on PATH. Activate the project env first." >&2
    exit 1
fi

echo "Creating W&B sweep from ${SWEEP_CFG} ..."

# `wandb sweep` writes its progress (including the `wandb agent <fq_id>`
# invocation hint we parse below) to stderr, so we redirect 2>&1 and tee a
# copy to the controlling terminal for visibility while capturing it.
SWEEP_OUTPUT=$(wandb sweep "${SWEEP_CFG}" 2>&1 | tee /dev/tty)
sweep_status=${PIPESTATUS[0]}
if (( sweep_status != 0 )); then
    echo "ERROR: 'wandb sweep' exited with status ${sweep_status}." >&2
    echo "       See output above for the underlying error (auth, schema, network)." >&2
    exit "${sweep_status}"
fi

# Extract the fully-qualified sweep ID (entity/project/id). Strip ANSI color
# codes first — newer wandb versions colorize stderr and the escape sequences
# break the awk-by-whitespace parse. The CLI also prints a bare ID on a
# separate line, but the FQ form is what `wandb agent` wants.
SWEEP_ID=$(echo "${SWEEP_OUTPUT}" \
    | sed -E 's/\x1b\[[0-9;]*m//g' \
    | grep -oE 'wandb agent [^[:space:]]+' \
    | tail -n1 \
    | awk '{print $3}')

if [[ -z "${SWEEP_ID}" ]]; then
    echo "ERROR: failed to parse sweep ID from wandb output." >&2
    echo "       Check the output above and submit manually:" >&2
    echo "         SWEEP_ID=<entity/project/id> sbatch sweep_agent.sbatch" >&2
    exit 1
fi

cat <<EOF

Sweep created: ${SWEEP_ID}

To launch SLURM agents (10-job array, 4 concurrent, 8 trials each):

    SWEEP_ID=${SWEEP_ID} sbatch sweep_agent.sbatch

To customize the array size or trials per job:

    SWEEP_ID=${SWEEP_ID} TRIALS_PER_JOB=4 sbatch --array=0-19%6 sweep_agent.sbatch

Watch progress:  https://wandb.ai/${SWEEP_ID%/*}/sweeps/${SWEEP_ID##*/}
EOF
