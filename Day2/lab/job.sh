#!/bin/bash
#SBATCH -A ICT23_SMR3872
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time 00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=490000MB

module purge
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8

mpicc -O3 my_pi.c

mpirun -np 4 ./a.out
