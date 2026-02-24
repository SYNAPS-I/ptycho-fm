#!/bin/bash
set -x
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# path to output directory
OUTPUT="/pscratch/sd/s/shas1693/data/ptycho"

module load pytorch

echo "Enabling profiling..."
NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
PROFILE_DIR="$OUTPUT/profiles"
mkdir -p "$PROFILE_DIR"
export PROFILE_CMD="nsys profile $NSYS_ARGS -o $PROFILE_DIR/vit-profile"

# Run command
cmd="$PROFILE_CMD python main.py"

nodes=1
ngpu=4 # number of GPUs (single node)
srun -u \
  -N $nodes \
  --ntasks-per-node $ngpu \
  --cpus-per-task=32 \
  --gpus-per-node $ngpu \
    bash -c "$cmd"

