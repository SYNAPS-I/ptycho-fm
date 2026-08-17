#!/bin/bash
JOB_SCRIPT="job_submission_nersc.sh"

CONFIGS=(
  # "configs/e192_d6.yaml"
 "configs/e256_d8.yaml"
 "configs/e384_d8.yaml"
#  "configs/e512_d8.yaml"
#  "configs/e512_d12.yaml"
#  "configs/e640_d12.yaml"
#  "configs/e768_d12.yaml"
#  "configs/e768_d16.yaml"
#  "configs/e1024_d12.yaml"
#  "configs/e1024_d16.yaml"
#  "configs/e1024_d24.yaml"
#  "configs/e1536_d16.yaml"
#  "configs/e1536_d24.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "Missing config: $cfg" >&2
    exit 1
  fi

  sbatch "$JOB_SCRIPT" "$cfg"
done
