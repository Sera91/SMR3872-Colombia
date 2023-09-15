from cuml.dask.cluster import KMeans
from cuml.dask.datasets import make_blobs
import sys
from dask.distributed import Client

sched_file = str(sys.argv[1])
nworkers = int(sys.argv[2])

c = Client(scheduler_file=sched_file)
print("client information ",c)

# 2. Blocks until num_workers are ready
print("Waiting for " + str(nworkers) + " workers...")
c.wait_for_workers(n_workers=nworkers)

n_centers = 5

X, _ = make_blobs(n_samples=10000, centers=n_centers)

k_means = KMeans(n_clusters=n_centers)
k_means.fit(X)

labels = k_means.predict(X)#!/bin/bash

#SBATCH --job-name=COOL_JOB_NAME    # create a short name for your job
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=240GB     
