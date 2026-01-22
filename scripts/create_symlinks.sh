#!/usr/bin/env bash
# Create symlinks for fine-tuning data.
# Flattens a nested directory structure into a single directory containing:
#   <scan>_dp.hdf5 and <scan>_para.hdf5

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") <base_dir> <dest_dir>

Environment overrides:
  BASE_DIR   Base directory (if arg1 omitted)
  DEST_DIR   Destination directory (if arg2 omitted)
  DP_NAME    Diffraction file name inside each scan dir (default: ptychodus_dp.hdf5)
  PARA_NAME  Parameter file name inside each scan dir (default: ptychodus_para.hdf5)
EOF
}

BASE="${1:-${BASE_DIR:-}}"
DEST="${2:-${DEST_DIR:-}}"
DP_NAME="${DP_NAME:-ptychodus_dp.hdf5}"
PARA_NAME="${PARA_NAME:-ptychodus_para.hdf5}"

if [[ -z "$BASE" || -z "$DEST" ]]; then
  usage
  exit 2
fi

# Check if base directory exists
if [[ ! -d "$BASE" ]]; then
    echo "ERROR: Base directory does not exist: $BASE"
    exit 1
fi

# Create destination directory
mkdir -p "$DEST"
echo "Created destination directory: $DEST"

# Counter for progress
count=0
skipped=0

# Loop through range directories (S02000-02999, S03000-03999, etc.)
for range_dir in "$BASE"/S*-*; do
    if [[ ! -d "$range_dir" ]]; then
        continue
    fi

    echo "Processing $(basename "$range_dir")..."

    # Loop through scan directories (S02252, S02253, etc.)
    for scan_dir in "$range_dir"/S[0-9]*; do
        if [[ ! -d "$scan_dir" ]]; then
            continue
        fi

        scan=$(basename "$scan_dir")
        dp_file="$scan_dir/$DP_NAME"
        para_file="$scan_dir/$PARA_NAME"

        # Check if both files exist
        if [[ -f "$dp_file" && -f "$para_file" ]]; then
            ln -sf "$dp_file" "$DEST/${scan}_dp.hdf5"
            ln -sf "$para_file" "$DEST/${scan}_para.hdf5"
            count=$((count + 1))
        else
            echo "  WARNING: Missing files in $scan_dir"
            skipped=$((skipped + 1))
        fi
    done
done

echo ""
echo "========================================"
echo "Done!"
echo "Created symlinks for $count scans"
echo "Skipped $skipped directories (missing files)"
echo "Output directory: $DEST"
echo ""
echo "Verify with: ls -la $DEST | head -20"
echo "Total files: $(ls -1 "$DEST" | wc -l)"
echo "========================================"
