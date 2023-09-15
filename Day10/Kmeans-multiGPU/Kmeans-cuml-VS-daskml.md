Today we are able to run different classical ML resources 


RAPIDS Release 0.9 introduced two new multi-node, multi-GPU (MNMG) algorithms to cuML, the RAPIDS machine learning library. Random forests and k-means have been given the ability to scale up & out with the Dask distributed computing library. In this blog, I will focus on how we scaled our k-means algorithm, the first algorithm to fully utilize our new scalable architecture (which I will be explaining in detail in an upcoming blog). By the end of this blog, I hope you are as excited about the future of scalable and performant machine learning as we are!


#Kmeans multi-gpu 

The k-means algorithm contains two basic stages, with an optional third stage.

  - Choose an initial set of k cluster centroids
  - Iteratively update the centroids until convergence or some max number of iterations is reached.
  - Optional: Use the centroids to predict on unseen data points

K-means is sensitive to the initial choice of centroids. A bad initialization may never produce good results. Since finding the optimal initialization is very costly and hard to verify, we generally use heuristics. Aside from random choice, cuML’s k-means provides the scalable k-means++ (ork-means||) initialization method, which is an efficient parallel version of the inherently sequential kmeans++ algorithm.


K-means is used by businesses to segment users and/or customers based on various attributes and behavioral patterns. It also finds use in anomaly detection, fraud detection, and document clustering.

The cuML project contains a C++ library with a growing collection of both dense and sparse CUDA primitives for machine learning. These GPU-accelerated primitives provide building blocks, such as linear algebra and statistics, for computations on feature matrices. Many algorithms in cuML, such as k-means, are constructed from these primitives.


We benchmarked the single-GPU k-means implementation against scikit-learn so we could determine the impact of cuML’s GPU acceleration. While scikit-learn is able to parallelize k-means using multiple CPU cores (by setting the n_jobs argument to -1), the GPU k-means implementation continues to demonstrate better performance as the data sizes are increased.

I ran this benchmark on a single NVIDIA GPU (32GB GV100) on an NVIDIA DGX1. The scikit-learn benchmark was taken on the same DGX1, using all 40-cores (80 total threads) of its 2.20GHz Intel Xeon CPU E5–2698 v4. We are seeing more than 100x speedup as the number of data samples reaches into the millions.
