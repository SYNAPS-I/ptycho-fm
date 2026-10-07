#!/usr/bin/env bash
# Run this inside an interactive PBS/qsub session, for example:
#   qsub -I -q AiLowQ -l mem=400mb,ncpus=1,ngpus=1,gdata=true
#
# The script flattens the nested ptychodus data layout using symlinks, updates
# config.yaml to point at the flat directory, then starts training.

# WARNING: Not tested with the current status of Califone

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/config.yaml}"

# This is the parent that contains S03000-03999. create_symlinks.sh scans
# "$RAW_BASE"/S*-*/Sxxxxx for ptychodus_dp.hdf5 and ptychodus_para.hdf5.
RAW_BASE="${RAW_BASE:-/gdata/dm/CAI/synaps_i/data/converted/lamni/2025-3/31ide_2025-11-18/SparkPixRT2_130nm_rerecon/eiger_4}"

# Directory used by training. It will contain Sxxxxx_dp.hdf5 and
# Sxxxxx_para.hdf5 symlinks.
FLAT_DIR="${FLAT_DIR:-/scratch/aileenluo/ptycho-vit/flat_data/S03000-03999}"

# Normalization dictionary generated from the configured training data path.
NORMALIZATION_PATH="${NORMALIZATION_PATH:-$FLAT_DIR/normalization.pkl}"

# File names inside each nested scan directory.
export DP_NAME="${DP_NAME:-ptychodus_dp.hdf5}"
export PARA_NAME="${PARA_NAME:-ptychodus_para.hdf5}"

# Set to 0 if your interactive environment already has the right Python stack.
LOAD_MODULES="${LOAD_MODULES:-1}"

# Repository virtual environment. Set VENV_PATH to override, or
# REQUIRE_VENV=0 to fall back to whatever python is on PATH.
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"
REQUIRE_VENV="${REQUIRE_VENV:-1}"

# Set to 0 if config.yaml already points at FLAT_DIR and should not be edited.
UPDATE_CONFIG="${UPDATE_CONFIG:-1}"

# Training launch settings. Keep this in sync with the GPU count requested from
# qsub, for example ngpus=4 with NPROC_PER_NODE=4.
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

cd "$PROJECT_DIR"

if [[ -z "${PBS_JOBID:-}" ]]; then
  echo "WARNING: PBS_JOBID is not set. This script is intended to run inside qsub -I."
fi

if [[ "$LOAD_MODULES" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load pytorch/2.6.0
fi

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate"
elif [[ "$REQUIRE_VENV" == "1" ]]; then
  echo "ERROR: Repository venv not found at $VENV_PATH"
  echo "Set VENV_PATH=/path/to/venv or REQUIRE_VENV=0 to use python from PATH."
  exit 1
else
  echo "WARNING: Repository venv not found at $VENV_PATH; using python from PATH."
fi

echo "Project: $PROJECT_DIR"
echo "Config:  $CONFIG_PATH"
echo "Source:  $RAW_BASE"
echo "Flat:    $FLAT_DIR"
echo "Norm:    $NORMALIZATION_PATH"
echo "Python:  $(command -v python)"
echo "Launch:  torchrun --nnodes $NNODES --nproc-per-node $NPROC_PER_NODE main.py"
echo

bash "$PROJECT_DIR/scripts/create_symlinks.sh" "$RAW_BASE" "$FLAT_DIR"

if [[ "$UPDATE_CONFIG" == "1" ]]; then
  BACKUP_PATH="${CONFIG_PATH}.bak.$(date +%Y%m%d_%H%M%S)"
  cp "$CONFIG_PATH" "$BACKUP_PATH"
  echo "Backed up config.yaml to $BACKUP_PATH"

  python - "$CONFIG_PATH" "$FLAT_DIR" "$NORMALIZATION_PATH" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
flat_dir = sys.argv[2]
normalization_path = sys.argv[3]

with config_path.open("r") as f:
    config = yaml.safe_load(f)

old_path = config.setdefault("data", {}).get("data_path")
old_norm_path = config["data"].get("normalization_dict_path")
old_test_norm_path = config["data"].get("test_normalization")
config["data"]["data_path"] = flat_dir
config["data"]["normalization_dict_path"] = normalization_path
config["data"]["test_normalization"] = normalization_path

with config_path.open("w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print(f"Updated data.data_path: {old_path} -> {flat_dir}", flush=True)
print(
    "Updated normalization paths: "
    f"{old_norm_path}, {old_test_norm_path} -> {normalization_path}",
    flush=True,
)
PY
else
  echo "Skipping config.yaml update because UPDATE_CONFIG=$UPDATE_CONFIG"
fi

# Normalization only needs to be built once for a persistent FLAT_DIR.
# Uncomment this block if the HDF5 set changes or normalization.pkl is missing.
# echo
# echo "Creating normalization dictionary..."
# python "$PROJECT_DIR/scripts/make_normalization_dict.py" \
#   --config "$CONFIG_PATH" \
#   --output "$NORMALIZATION_PATH" \
#   --update-config

echo
echo "Starting training..."
python -m torch.distributed.run \
  --nnodes "$NNODES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  main.py
