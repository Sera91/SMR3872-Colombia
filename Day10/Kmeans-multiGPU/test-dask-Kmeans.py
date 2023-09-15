from dask_ml.cluster import KMeans
from sklearn.datasets import make_blobs
from dask.distributed import Client

sched_file = str(sys.argv[1])
nworkers = int(sys.argv[2])

c = Client(scheduler_file=sched_file)
print("client information ",c)

# 2. Blocks until num_workers are ready
print("Waiting for " + str(nworkers) + " workers...")
c.wait_for_workers(n_workers=nworkers)

n_centers = 5

n_centers = 5

X, _ = make_blobs(n_samples=10000, centers=n_centers)

k_means = KMeans(n_clusters=n_centers)
k_means.fit(X)

labels = k_means.predict(X)
