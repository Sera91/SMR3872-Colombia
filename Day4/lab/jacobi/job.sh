#!/bin/bash
#SBATCH -A ICT23_SMR3872
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time 00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=490000MB

#modulefiles to be loaded to have MPI on Leonardo
module purge
module load nvhpc

cd serial
make c
cd ..

./serial/laplace2d_serial


