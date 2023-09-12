import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from kPCA import rbf_kernel_pca
import numpy as np

#If you want to see a detailed description of kernel-PCA
#you can look a the webpage
#https://sebastianraschka.com/Articles/2014_kernel_pca.html


X, y = make_moons(n_samples=100, random_state=123)

plt.figure(figsize=(8,6))
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', alpha=0.5)
plt.title('A nonlinear 2Ddataset')
plt.ylabel('y coordinate')
plt.xlabel('x coordinate')
plt.savefig('moons_original_space.png')
plt.show()


X_pc = rbf_kernel_pca(X, gamma=15, k_components=2)

plt.figure(figsize=(8,6))
plt.scatter(X_pc[y==0, 0], X_pc[y==0, 1], color='red', alpha=0.5)
plt.scatter(X_pc[y==1, 0], X_pc[y==1, 1], color='blue', alpha=0.5)
plt.title('First 2 principal components after RBF Kernel PCA')
plt.text(-0.18, 0.18, 'gamma = 15', fontsize=12)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('2D_kPCA.png')
plt.show()



#if we project the input data points along the first PC we get the following representation
plt.figure(figsize=(8,6))
plt.scatter(X_pc[y==0, 0], np.zeros((50)), color='red', alpha=0.5)
plt.scatter(X_pc[y==1, 0], np.zeros((50)), color='blue', alpha=0.5)
plt.title('First principal component after RBF Kernel PCA')
plt.text(-0.19, 0.18, 'gamma = 15',fontsize=12)
plt.xlabel('PC1')
plt.savefig('1D_kPCA.png')
plt.show()


