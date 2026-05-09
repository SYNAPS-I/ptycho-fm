#!/bin/bash

image=registry.nersc.gov/amsc006/shas1693/ptychofm:26.01
srun -N 4 -n 250 --mpi=pmi2 --module=gpu shifter --image=${image} bash -c "
    python pack_hdf5.py
"