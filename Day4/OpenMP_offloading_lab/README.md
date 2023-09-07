# OpenMP GPU offload exercise.

*SAXPY*: Single Precision a * X + Y, where "a" is a scalar and X and Y vectors.

Compile the code using:

```
nvc -mp=gpu -gpu=cc80 -O3 -o SAXPY.x saxpy.c
```

In Leonardo HPC cluster load `nvc` with the command:

```
module load nvhpc/23.1
```

Try to avoid using the file in examples, before creating your own version, unless it is completely necessary!


