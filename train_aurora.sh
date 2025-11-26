export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR

NNODES=1
NGPUSPNODE=12
NGPUS=$((NNODES * NGPUSPNODE))
echo $NGPUS
mpiexec -n $NGPUS -ppn $NGPUSPNODE python main.py --use-random-data
