#!/bin/bash
#SBATCH -J job_name
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 256
#SBATCH -c 8                    
#SBATCH -t 04:00:00
#SBATCH -A amsc006_g
#SBATCH -o exp1_%j.log
#SBATCH -e exp1_%j.err

# --- Modules ---
module load pytorch/2.6.0
module load nccl/2.18.3-cu12

# --- Project location ---
PROJECT_DIR=/pscratch/sd/e/edey/ptycho-vit/
cd "$PROJECT_DIR"

# --- NCCL / runtime env ---
unset NCCL_DEBUG
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_NET_GDR_LEVEL=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 #8 before
export FI_CXI_DISABLE_CQ_HUGETLB=1
export NCCL_CROSS_NIC=0

# --- Rendezvous (env://) ---
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29500

# --- Config path (script must accept --config) ---
# CFG="config.yaml"

# --- Launch (128 nodes × 4 tasks/node = 512 GPUs) ---
srun -N 4 --ntasks-per-node=4 --ntasks=16 --gpus-per-task=1 --gpu-bind=none -l -u \
    python -m ptycho_vit.train

    # --config "$CFG"
