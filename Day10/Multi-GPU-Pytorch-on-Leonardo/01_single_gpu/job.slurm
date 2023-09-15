#!/bin/bash
#SBATCH -A ict23_smr3872
#SBATCH -p boost_usr_prod
#SBATCH --job-name=mnist         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
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

kernprof -l mnist_classify.py --epochs=3
