#!/bin/bash
source ~/.bashrc
export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR

cd /eagle/datascience/vsastry/projects/pytcho_vit/ptycho-vit
module use /soft/modulefiles
module load conda
conda activate base

NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))"

mpiexec -n $NGPUS -ppn $NGPU_PER_HOST python main.py --use-random-data > output_${NGPUS}.log 2>&1
