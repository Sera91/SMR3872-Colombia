import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh


def rbf_kernel_pca(X, gamma, k_components):    
    """
    RBF kernel PCA implementation.        
    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_examples, n_features]      
    gamma: float  tuning parameter of the RBF kernel        
    k_components: int  number of principal components to return        
    Returns
    ------------
    X_pc: {NumPy ndarray}, shape = [n_examples, k_features]     Projected dataset       
    """    
    # Calculate pairwise squared Euclidean distances    
    # in the MxN dimensional dataset.    
    sq_dists = pdist(X, 'sqeuclidean')        
    # Convert pairwise distances into a square matrix.    
    mat_sq_dists = squareform(sq_dists)        
    # Compute the symmetric kernel matrix.    
    K = exp(-gamma * mat_sq_dists)        
    # Center the kernel matrix.    
    N = K.shape[0]    
    one_n = np.ones((N,N))/N    
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)        
    # Obtaining eigenpairs from the centered kernel matrix    
    # scipy.linalg.eigh returns them in ascending order    
    eigvals, eigvecs = eigh(K)    
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]        
    # Collect the top k eigenvectors (projected examples)    
    X_pc = np.column_stack([eigvecs[:, i] for i in range(k_components)])        
    return X_pc

