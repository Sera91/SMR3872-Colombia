#!/bin/bash
#SBATCH -A ict23_smr3872
#SBATCH -p boost_usr_prod
#SBATCH --job-name=torchlight        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=sdigioia@sissa.it
#SBATCH -e run.err
#SBATCH -o run.out

module purge
module load gcc
module load cuda
module load openmpi
source $HOME/.bashrc
conda activate /leonardo_work/ICT23_SMR3872/shared-env/Gabenv

#export PL_TORCH_DISTRIBUTED_BACKEND=gloo

srun python myscript.py
