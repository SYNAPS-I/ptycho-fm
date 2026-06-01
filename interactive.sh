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
image=registry.nersc.gov/amsc006/shas1693/ptychofm:26.01

echo "Enabling profiling..."
NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
PROFILE_DIR="$OUTPUT/profiles"
mkdir -p "$PROFILE_DIR"
export PROFILE_CMD="nsys profile $NSYS_ARGS -o $PROFILE_DIR/iter"

# Run command
cmd="$PROFILE_CMD python main_iters.py --config config.yaml"

nodes=4
ngpu=4 # number of GPUs (single node)
srun -u --mpi=pmi2 \
  -N $nodes \
  --ntasks-per-node $ngpu \
  --cpus-per-task=32 \
  --gpus-per-node $ngpu \
  shifter --image=$image --module=gpu,nccl-cu13-plugin bash -c "$cmd"

