#!/bin/bash
# Submit explicitly selected model/site configs from the repository root.
set -euo pipefail
if [ "$#" -eq 0 ]; then
    echo "Usage: bash hpc_submission_scripts/submit_isoflop.sh CONFIG [CONFIG ...]" >&2
    exit 2
fi
for cfg in "$@"; do
    if [[ ! -f "$cfg" ]]; then
        echo "Missing config: $cfg" >&2
        exit 1
    fi
    sbatch hpc_submission_scripts/nersc.sh "$cfg"
done
