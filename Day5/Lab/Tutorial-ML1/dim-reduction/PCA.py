import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#3rd Exercise of Practice1
#Download the Ionosphere dataset, containing info on radar
#returns from the ionosphere, from the website:
#http://archive.ics.uci.edu/ml/datasets/Ionosphere
#and perform a PCA analysis of the Ionosphere dataset

#part of 4th Exercise
#create routine to estimate the Euclidean distance among datapoins
#in the PCA space

#**********ROUTINES****************************************

#personal PCA algorithm
def my_PCA(X_standard , num_components):
     
    
    #Step-2  Calculate covariance matrix:
    cov_mat = np.cov(X_standard , rowvar = False)
     
    #Step-3  Estimate eigenvalues and eigenvectors associated to the Covariance mat.
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

    
    # Plotting the correlation matrix  
    plt.figure(figsize=(10,10))
    sns.heatmap(cov_mat, vmax=1, square=True,annot=True)
    plt.title('Correlation matrix')
    plt.show()
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5 : selecting more important features
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6 : Projecting the points in the PC space 
    X_reduced = np.dot(eigenvector_subset.transpose() , X_standard.transpose() ).transpose()
     
    return X_reduced, sorted_eigenvalue , sorted_eigenvectors



# plotting in 2D
def plot_scatter(pc1, pc2, classes, figname):
    fig, ax = plt.subplots(figsize=(15, 8))
    
    classes_unique = list(set(classes))
    classes_colors = ["r","b"]
    
    for i, spec in enumerate(classes):
        plt.scatter(pc1[i], pc2[i], label = spec, s = 20, c=classes_colors[classes_unique.index(spec)])
        ax.annotate(str(i+1), (pc1[i],pc2[i]))
    
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 15}, loc=4)
    
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.axhline(y=0, color="grey", linestyle="--")
    ax.axvline(x=0, color="grey", linestyle="--")
    plt.grid()
    #plt.axis([-4, 4, -3, 3])
    plt.savefig(figname)
    plt.clf()
    return




def pca_lambda_plot(eigvals):
    lambda_eigenvals = np.argsort(eigvals)[::-1] 
    eigvals = eigvals[lambda_eigenvals]# compute eigenvecs
    plt.semilogy(lambda_eigenvals, eigvals, 'ob')
    plt.title("PC eigenvalues")
    plt.xlabel(r"$\lambda$ number")
    plt.ylabel(r"$\lambda$")
    plt.savefig("lambda_plot_iono_data.pdf")
    plt.show()
    return 



def Euclidean_dist(x_i, x_j):
    dx = x_i - x_j
    return np.sqrt(np.dot(dx, dx))


#**************************READING INPUT DATA***************************************


df_iono = pd.read_csv("ionosphere-dataset/ionosphere.data")


cols =  [ f'real_{i//2}' if i%2==0 else f'imaginary_{i//2}' for i in range(0,34)] + ['target']
df_iono.columns = cols

df_iono.describe()

#convert dataframe into numpy matrix
iono_arr = df_iono.iloc[:,:-1].to_numpy()



#dropping the target  column  (bad/good)
classes = df_iono["target"].tolist()
X = df_iono.drop("target", 1)






#sys.exit()


#PCA with scikit learn
pca_iono = PCA(n_components=2)

#STEP 1: Standardizing the data

X_standard = StandardScaler().fit_transform(X)


principalComponents_iono = pca_iono.fit_transform(X_standard)

pc1_scikit= - principalComponents_iono[:,0]
pc2_scikit= - principalComponents_iono[:,1]
plot_scatter(pc1_scikit, pc2_scikit, classes, 'plot_scatter_PC_scikit.pdf')
print('Explained variation per principal component: {}'.format(pca_iono.explained_variance_ratio_))


#PCA with my implementation


#Step-1 : data standardization

X_standard_my = (X - X.mean()) / X.std(ddof=0)

X_standard_arr = X_standard_my.to_numpy()

X_standard_arr[np.isnan(X_standard_arr)] = 0.0

#X_meaned = (X - np.mean(X , axis = 0))   # subtract mean from each observation 

#testing my standardization procedure :

print('maximum difference found between my standardized dataset and the one standardized with scikit is:')
print(np.max(X_standard_arr - X_standard))


myX_reduced, eigvals, eigvectors = my_PCA(X_standard_arr, 2)

my_pc1= - myX_reduced[:,0]
my_pc2= myX_reduced[:,1]



#Scatter plot of datapoints projected in the PC space
plot_scatter(my_pc1, my_pc2, classes, 'plot_scatter_PC_mine.pdf')



#Plotting the sorted eigenvalues
pca_lambda_plot(eigvals)



# plotting the variance explained by each PC to see how many principal components consider to treat this dataset
explained_variance=(eigvals / np.sum(eigvals))*100
plt.figure(figsize=(8,4))
plt.bar(range(len(explained_variance)), explained_variance, alpha=0.6)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Dimensions')
plt.show()



# Estimating Euclideand distance in PCA space



df_iono_PC = pd.DataFrame( {'PC-1': -myX_reduced[:,0],
                            'PC-2': myX_reduced[:,1],
                            'target': np.array(classes),
                                     })


df_iono_reduced= df_iono_PC[['PC-1','PC-2']]
point_0 = df_iono_reduced.iloc[0].values
point_1 = df_iono_reduced.iloc[1].values

#testing Euclidean distance:
Myeuclide_dist_pair1= Euclidean_dist(point_0, point_1)
from scipy.spatial import distance
Scipy_dist_pair1=distance.euclidean(point_0, point_1)

test_dist_res='not-passed'
if abs(Myeuclide_dist_pair1-Scipy_dist_pair1)< 1.e-15:
   test_dist_res='passed'

print('test distance: ', test_dist_res)
Euclidean_distance_arr = np.array([Euclidean_dist(df_iono_reduced.iloc[0].values, df_iono_reduced.iloc[i].values) for i in range(df_iono_reduced.shape[0])])

sys.exit()










