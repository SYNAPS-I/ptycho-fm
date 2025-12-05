#!/bin/bash
source ~/.bashrc
export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR

cd /flare/datascience/vsastry/projects/ptcho_vit/ptycho-vit 
module load frameworks 
source /flare/datascience/vsastry/projects/ptcho_vit/venvs/pytcho_vit/bin/activate 
NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=12
NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))"
#export OMP_NUM_THREADS=4
#export OMP_PLACES=threads
#unset CCL_OP_SYNC
#export CCL_OP_SYNC=0

#export OMP_PLACES=cores
#export OMP_PROC_BIND=close
#export OMP_NUM_THREADS=8

## Option 1
export CPU_BINDING1="list:4:9:14:19:20:25:56:61:66:71:74:79" # 12 ppn to 12 cores
## Option 2
export CPU_BINDING2="list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79" # 12 ppn with each rank having 4 cores
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7,8,9,10,11
## Option 1 for oneCCL worker affinity 
#export CCL_WORKER_AFFINITY=42,43,44,45,46,47,94,95,96,97,98,99
export CPU_BINDING3="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
## Option 2
#unset CCL_WORKER_AFFINITY  # Default will pick up from the last 24 cores even if you didn't specify these in the binding.
#EXT_ENV="--env FI_CXI_DEFAULT_CQ_SIZE=1048576"

#mpiexec -n $NGPUS -ppn $NGPU_PER_HOST --cpu-bind  $CPU_BINDING3 python main.py --use-random-data > output_${NGPUS}.log 2>&1
mpiexec -n $NGPUS -ppn $NGPU_PER_HOST --cpu-bind  $CPU_BINDING3 python main.py > output_rawdata_workers8_${NGPUS}.log 2>&1
#mpiexec -n $NGPUS -ppn $NGPU_PER_HOST -- iprof -- python main.py --use-random-data > output_${NGPUS}.log 2>&1
