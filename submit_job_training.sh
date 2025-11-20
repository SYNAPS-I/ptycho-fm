#!/bin/bash
#SBATCH --job-name=ptycho_vit_training
#SBATCH --account=m5073
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --output=ptycho_vit_training_%j.out
#SBATCH --error=ptycho_vit_training_%j.err

# Load Python module
module load python/3.11

# Navigate to project directory
cd /global/cfs/cdirs/m5073/pecomyint/ptycho-vit

# Activate virtual environment (for main shell environment)
source .venv/bin/activate

# Set NCCL environment variables (from multinode.sh approach)
# These help NCCL work correctly with SLURM's GPU assignment
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=^lo,docker0  # Use network interfaces (exclude loopback)
export NCCL_NET_GDR_LEVEL=2  # Enable GPU Direct RDMA
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Better error handling (updated from deprecated NCCL_ASYNC_ERROR_HANDLING)
export NCCL_CROSS_NIC=0  # Disable cross-NIC communication
unset NCCL_DEBUG  # Reduce debug output (set to INFO if needed)

# Run with multiple GPUs across multiple nodes using srun (like ptycho_simulation_factory)
# 2 nodes × 4 GPUs = 8 GPUs total (8 tasks)
# SLURM sets SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID automatically
# The Python code will map these to RANK, WORLD_SIZE, LOCAL_RANK for PyTorch distributed
# MASTER_ADDR and MASTER_PORT are set automatically in main.py from SLURM_JOB_NODELIST
# Using full path ensures venv's site-packages are found
# Use --gpu-bind=none like multinode.sh to avoid GPU binding issues
srun --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=1 --gpu-bind=none \
     /global/cfs/cdirs/m5073/pecomyint/ptycho-vit/.venv/bin/python main.py

