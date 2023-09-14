#!/bin/bash -x

module purge
module load hdf5/1.12.2--openmpi--4.1.4--nvhpc--23.1

C_FLAGS="-O3"
mpicc ${C_FLAGS} -c benchmark_LBE3D_IO.c -I/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/nvhpc-23.1/hdf5-1.12.2-6a3ddj2hipc7tm4xrdqh4ijifigxq3bq/include
mpicc ${C_FLAGS} -o benchmark_LBE3D_IO.x benchmark_LBE3D_IO.o -L/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/nvhpc-23.1/hdf5-1.12.2-6a3ddj2hipc7tm4xrdqh4ijifigxq3bq/lib -lhdf5
