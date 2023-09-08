#!/bin/bash
#SBATCH -A ict23_esp_0
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=60G
#SBATCH --ntasks-per-node=32 # out of 128
#SBATCH --gres=gpu:4          # 1 gpus per node out of 4
#SBATCH --job-name=cl_test
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=sdigioia@sissa.it
#SBATCH -o $SLURM_SUBMIT_DIR/run.out
#SBATCH -e $SLURM_SUBMIT_DIR/run.err

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8 

cd $SLURM_SUBMIT_DIR
source $HOME/.bashrc

conda activate /leonardo_work/ICT23_SMR3872/shared-env/MLenv/

python --version

