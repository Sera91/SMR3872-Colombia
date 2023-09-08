import pickle
import cuml
from cuml.common.device_selection import using_device_type
from cuml.common.device_selection import set_global_device_type, get_global_device_type
from cuml.neighbors import NearestNeighbors
from cuml.manifold import UMAP
from cuml.linear_model import LinearRegression
from cuml.datasets import make_regression, make_blobs
from cuml.model_selection import train_test_split

X_blobs, y_blobs = make_blobs(n_samples=2000, n_features=20)
X_train_blobs, X_test_blobs, y_train_blobs, y_test_blobs = train_test_split(X_blobs, y_blobs, test_size=0.2, shuffle=True)

X_reg, y_reg = make_regression(n_samples=2000, n_features=20)
X_train_reg, X_test_reg, y_train_reg, y_tes_reg = train_test_split(X_reg, y_reg, test_size=0.2, shuffle=True)

nn = NearestNeighbors()
#with using_device_type('gpu'):
with using_device_type('cpu'):
    nn.fit(X_train_blobs)
    nearest_neighbors = nn.kneighbors(X_test_blobs)
