#!/bin/bash -l
#PBS -A SYNAPS-I
#PBS -q <your_queue>
#PBS -l select=32                       # nodes — match fraction: 10%=32, 20%=64, 40%=128, 60%=256, 100%=512
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -N ptycho_vit_aurora
#PBS -o /flare/SYNAPS-I/vsastry/logs/vit_${PBS_JOBID}.log
#PBS -e /flare/SYNAPS-I/vsastry/logs/vit_${PBS_JOBID}.err

# =============================================================================
# Aurora multi-node training with node-local tmpfs staging.
# Stages a fraction of the packed shards to $TMPDIR/packed on each node,
# then runs main.py with data.local_stage.enabled=true in config.yaml.
# =============================================================================

module load frameworks

PROJECT_DIR=/flare/SYNAPS-I/vsastry/projects/ptcho_vit/simple_pack_fm_ss/ptycho-vit
cd "$PROJECT_DIR"

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
unset OMP_PLACES
export HDF5_USE_FILE_LOCKING=FALSE
export FI_CXI_DISABLE_CQ_HUGETLB=1
export MPICH_GPU_SUPPORT_ENABLED=1

# Consistent TMPDIR across all ranks (PBS gives each job a unique one)
export TMPDIR=${TMPDIR:-/tmp}
export STAGE_DIR="$TMPDIR/packed"

NNODES=$(wc -l < "$PBS_NODEFILE")
NRANKS_PER_NODE=12                     # Aurora: 6 GPUs x 2 tiles
NTOTRANKS=$((NNODES * NRANKS_PER_NODE))

echo "=============================================="
echo "Aurora training — nodes=$NNODES ranks/node=$NRANKS_PER_NODE total=$NTOTRANKS"
echo "Staging to: $STAGE_DIR"
echo "=============================================="

# --- Stage shards to node-local tmpfs --------------------------------------
SOURCE_DIR=/flare/SYNAPS-I/simulated_data_cleanedProbe_2_packed
FRACTION=0.1                           # match config.yaml:data.local_stage.shard_fraction
SEED=8
WORKERS=4

echo "[job] staging ${FRACTION} of shards from ${SOURCE_DIR} -> ${STAGE_DIR}"
mpiexec -np ${NTOTRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind depth \
    python scripts/stage_pack_to_local.py \
        --source "$SOURCE_DIR" \
        --local-dir "$STAGE_DIR" \
        --fraction "$FRACTION" \
        --seed "$SEED" \
        --workers "$WORKERS"

STAGE_RC=$?
if [ $STAGE_RC -ne 0 ]; then
    echo "[job] staging failed (rc=$STAGE_RC) — aborting"
    exit $STAGE_RC
fi

# --- Train ----------------------------------------------------------------
echo "[job] launching training"
mpiexec -np ${NTOTRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind depth \
    python main.py --config config.yaml
