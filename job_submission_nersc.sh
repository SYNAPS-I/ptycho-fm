#!/bin/bash
#SBATCH -J ptycho
#SBATCH -q regular
#SBATCH -C "gpu&hbm80g"
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -A amsc006
#SBATCH --image=registry.nersc.gov/amsc006/shas1693/ptychofm:26.01
#SBATCH --module=gpu,nccl-cu13-plugin
#SBATCH -o ptycho_%j.log
#SBATCH -e ptycho_%j.err

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1
export NCCL_CROSS_NIC=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CONFIG_ARG=""
if [ $# -gt 0 ]; then
    CONFIG_ARG="--config $1"
fi

srun -u --mpi=pmi2 --module=gpu \
    shifter bash -c "
    python main_iters.py $CONFIG_ARG
"
