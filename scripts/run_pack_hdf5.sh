#!/bin/bash

image=registry.nersc.gov/amsc006/shas1693/ptychofm:26.01
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
srun -N 4 -n 250 --mpi=pmi2 --module=gpu \
    shifter --image="${image}" python "${SCRIPT_DIR}/pack_hdf5.py" "$@"
