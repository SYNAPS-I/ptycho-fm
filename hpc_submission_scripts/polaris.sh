#!/bin/bash -l
#PBS -A SYNAPS-I
#PBS -q demand
#PBS -l select=50
#PBS -l place=scatter
#PBS -l walltime=01:00:00  
#PBS -l filesystems=home:eagle
#PBS -o /eagle/SYNAPS-I/mingdu/ptycho-vit/workspace/ptycho_vit_polaris_%j.log
#PBS -e /eagle/SYNAPS-I/mingdu/ptycho-vit/workspace/ptycho_vit_polaris_%j.err
#PBS -N ptycho_vit_polaris

# =============================================================================
# Polaris Multi-Node Multi-GPU Training Script
# Polaris nodes have 4 NVIDIA A100 GPUs
# For PyTorch DDP: 1 rank per GPU = 4 ranks per node
# =============================================================================

# --- Modules ---
module use /soft/modulefiles
module load conda
conda activate

# --- Project location ---
PROJECT_DIR=/eagle/SYNAPS-I/mingdu/ptycho-vit/
cd "$PROJECT_DIR"

# --- Fix for multiprocessing socket path issue ---
export TMPDIR=/tmp

# --- Runtime environment ---
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
unset OMP_PLACES
export WANDB_API_KEY=$(cat /eagle/SYNAPS-I/mingdu/api_keys/wandb.txt)

# --- NCCL settings for PyTorch distributed on NVIDIA GPUs ---
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_DEBUG=WARN

# --- Fabric settings ---
export FI_CXI_DISABLE_CQ_HUGETLB=1

# --- Calculate distributed training parameters ---
NNODES=$(wc -l < "$PBS_NODEFILE")
NRANKS_PER_NODE=4  # 4 GPUs per Polaris node
NTOTRANKS=$((NNODES * NRANKS_PER_NODE))
TOTAL_GPUS=$NTOTRANKS

echo "=============================================="
echo "Polaris Multi-GPU Training"
echo "=============================================="
echo "Nodes: $NNODES"
echo "Ranks per node: $NRANKS_PER_NODE"
echo "Total ranks: $NTOTRANKS"
echo "Total GPUs: $TOTAL_GPUS"
echo "=============================================="

# --- Launch distributed training ---
# MPI handles rank coordination; Python code uses MPI broadcast for MASTER_ADDR
mpiexec -np ${NTOTRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind depth \ #python -m ptycho_vit.train --config /eagle/SYNAPS-I/mingdu/ptycho-vit/workspace/models_for_FT/Fine_tune_360/config.yaml
    python -m ptycho_vit.train --config /eagle/SYNAPS-I/mingdu/ptycho-vit/workspace/models_for_FT/Fine_tune_360/config_res_ckpt.yaml
