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
#SBATCH -o /leonardo_work/ICT23_SMR3872/sdigioia/test_env/run.out
#SBATCH -e /leonardo_work/ICT23_SMR3872/sdigioia/test_env/run.err

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8 

cd /leonardo_work/ICT23_SMR3872/sdigioia/test_env/

source /leonardo/home/userexternal/sdigioia/.bashrc

#conda activate /m100_scratch/userexternal/sdigioia/geom2
#conda activate /m100_work/ICT23_ESP_C/env/GNNenv
conda activate /leonardo_work/ICT23_ESP_0/shared-env/MLenv

python --version

