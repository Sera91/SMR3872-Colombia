#!/bin/bash
#SBATCH -A ict23_smr3872
#SBATCH -p boost_usr_prod
#SBATCH --partition=gpu
#SBATCH --time 01:00:00
#SBATCH --nodes 3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --job-name dask_cuml_test
#SBATCH -o dask_cuml_test.out
#SBATCH -e dask_cuml_test.err

module load singularity
module load gcc
module load cuda

repodir=/leonardo_work/ICT23_SMR3872

SCHEDULER_DIR=$HOME/dask
WORKER_DIR=$HOME
SCRIPT=$repodir/test-cuml-Kmeans.py

if [ ! -d "$SCHEDULER_DIR" ]
then
    mkdir $SCHEDULER_DIR
fi

SCHEDULER_FILE=$SCHEDULER_DIR/my-scheduler.json

echo 'Running scheduler'
srun --nodes 1 --ntasks 1 --cpus-per-task 1 --gpus-per-task 1\
      singularity exec --nv $HOME/pytorch_23.05-py3.sif dask-scheduler --interface ib0 \
                     --scheduler-file $SCHEDULER_FILE \
                     --no-dashboard --no-show &

#Wait for the dask-scheduler to start
sleep 10

srun --nodes 2 --ntasks 4 --cpus-per-task 1  --gpus 4 --gpus-per-task 1 \
      singularity exec --nv $HOME/pytorch_23.05-py3.sif dask-cuda-worker --nthreads 1 --memory-limit 82GB --device-memory-limit 16GB --rmm-pool-size=15GB \
                       --interface ib0 --scheduler-file $SCHEDULER_FILE --local-directory $WORKER_DIR \
                       --no-dashboard &

#Wait for WORKERS
sleep 10

WORKERS=4

singularity exec --nv $HOME/pytorch_23.05-py3.sif python -u $SCRIPT $SCHEDULER_FILE $WORKERS

wait

#clean DASK files
rm -fr $SCHEDULER_DIR

echo "Done!"
