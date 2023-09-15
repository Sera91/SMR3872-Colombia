# Comparing cuML K-Means API Against Scikit-learn & Dask-ML
Today we are able to run different classical ML algorithm both on CPU/GPU in single node and multi-node configuration, thanks to libraries like Dask and RAPIDS.


RAPIDS make available tens of multi-node, multi-GPU (MNMG) algorithms to cuML, the RAPIDS machine learning library. Random forests and k-means have been given the ability to scale up & out with the Dask distributed computing library. In this blog, I will focus on the k-means algorithm, the first algorithm to fully utilize the RAPIDS scalable architecture. By the end, I hope you are as excited about the future of scalable and performant machine learning as I am!


#Kmeans multi-gpu 

The k-means algorithm contains two basic stages, with an optional third stage.

  - Choose an initial set of k cluster centroids
  - Iteratively update the centroids until convergence or some max number of iterations is reached.
  - Optional: Use the centroids to predict on unseen data points

K-means is sensitive to the initial choice of centroids. A bad initialization may never produce good results. Since finding the optimal initialization is very costly and hard to verify, we generally use heuristics. Aside from random choice, cuML’s k-means provides the scalable k-means++ (ork-means||) initialization method, which is an efficient parallel version of the inherently sequential kmeans++ algorithm.


K-means is used by businesses to segment users and/or customers based on various attributes and behavioral patterns. It also finds use in anomaly detection, fraud detection, and document clustering.

The cuML project contains a C++ library with a growing collection of both dense and sparse CUDA primitives for machine learning. These GPU-accelerated primitives provide building blocks, such as linear algebra and statistics, for computations on feature matrices. Many algorithms in cuML, such as k-means, are constructed from these primitives.


We benchmarked the single-GPU k-means implementation against scikit-learn so we could determine the impact of cuML’s GPU acceleration. While scikit-learn is able to parallelize k-means using multiple CPU cores (by setting the n_jobs argument to -1), the GPU k-means implementation continues to demonstrate better performance as the data sizes are increased.

I report here the benchmark on a single NVIDIA GPU (32GB GV100), NVIDIA DGX1. The scikit-learn benchmark was taken on the same DGX1, using all 40-cores (80 total threads) of its 2.20GHz Intel Xeon CPU E5–2698 v4. We are seeing more than 100x speedup as the number of data samples reaches into the millions.

First, a quick code example of K-Means in Scikit-learn
```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

n_centers = 5

X, _ = make_blobs(n_samples=10000, n_centers=n_centers)

k_means = KMeans(n_clusters = n_centers)
k_means.fit(X)

labels = k_means.predict(X)
```

To use cuML's Single-GPU API, we just change the imports
```python
from cuml.cluster import KMeans
from cuml.datasets import make_blobs

n_centers = 5

X, _ = make_blobs(n_samples=10000, n_centers=n_centers)

k_means = KMeans(n_clusters=n_centers)
k_means.fit(X)

labels = k_means.predict(X)
```

To use KMeans in Dask-ML, which is CPU-based, we just need to create a Dask `Client`
```python
from dask_ml.cluster import KMeans
from sklearn.datasets import make_blobs

from dask.distributed import Client
c = Client(<scheduler_address>)

n_centers = 5

X, _ = make_blobs(n_samples=10000, n_centers=n_centers)

k_means = KMeans(n_clusters=n_centers)
k_means.fit(X)

labels = k_means.predict(X)
```

And to use the multi-node multi-GPU API, we just change the imports again
```python
from cuml.dask.cluster import KMeans
from cuml.dask.datasets import make_blobs

from dask.distributed import Client
c = Client(<scheduler_address>)

n_centers = 5

X, _ = make_blobs(n_samples=10000, n_centers=n_centers)

k_means = KMeans(n_clusters=n_centers)
k_means.fit(X)

labels = k_means.predict(X)

```

To get a feel for cuML’s performance against a CPU-based algorithm in the Dask environment, I report below
the runtime for single-GPU (DGX) vs single-CPU run.

![single_node](https://github.com/Sera91/SMR3872-Colombia/blob/main/Day10/Kmeans-multiGPU/plots/single_node.png?raw=true)


# Multi-node multi-gpu benchmark

![multi_node](https://github.com/Sera91/SMR3872-Colombia/blob/main/Day10/Kmeans-multiGPU/plots/benchmark.png?raw=true)

