import os
from math import sqrt
import random
from random import seed
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
#from sklearn.cluster import KMeans
from operator import itemgetter


pd.options.display.max_rows = 4000



#1st Exercise of Practice2
# download "s3.txt" from http://cs.uef.fi/sipu/datasets/
# wget http://cs.uef.fi/sipu/datasets/s3.txt
#perform clustering analysis through Kmeans algorithm,
#using both standard centroid initialization and Kmeans-++ initialization
"""
K-means algorithm
given a set of points
1. select k centroids (with random/plus-plus initialization)
2. repeat EM point until tolerance is reached
 - label points (assigning each point to the nearest centroid)
 - update centroids
"""

##############################ROUTINES###############################################################

def read_input_file(path_to_file, filename, delimiter=","):
    input_data = np.genfromtxt(os.path.join(path_to_file, filename), delimiter=delimiter, dtype=float)
    print(input_data[0:3])
    print(len(input_data[:,0]))
    return input_data


def distance(data_points, centroids, N_clusters):
    N_points = len(data_points[:,0])
    print(N_points)
    dist_matrix =np.zeros((N_points, N_clusters), dtype=float)
    #distances_arr = []
    for i, centroid in enumerate(centroids):
            dist_matrix[:,i] = np.sqrt((data_points[:,0] -centroid[0])**2 + (data_points[:,1]-centroid[1])**2)
    return dist_matrix


    #questo si puo fare ancge con
    #np.argmin(np.linalg.norm(np.tile(centroids, (data_points.shape[0],1,1))- data_points.reshape(data_points.shape[0],1, data_points.shape[1])),axis=2),axis=1)


def guess_centroids(dataset, k):
    indices_points=range(len(dataset[:,0]))
    indices_centroids=random.sample(indices_points, k=k)
    return dataset[indices_centroids]



def guess_centroids_plus_plus(dataset, k):
    """K-means plus-plus initialization
       - step 1: choose the first cluster center (centroid) at random
       - step 2: assign the second centroid to the farthest point from the previous cluster centroid
       - step 3: assign the other centroids as the points at the max distance respect to the centroids
       previously identified"""
    indices_points=range(len(dataset[:,0]))
    indices_centroids=np.zeros(k, dtype=np.int64)
    index_first_centroid=random.sample(indices_points, k=1)[0]
    indices_centroids[0]= index_first_centroid
    first_dist_arr = np.sqrt((dataset[:,0] -dataset[index_first_centroid][0])**2 + (dataset[:,1]- dataset[index_first_centroid][1])**2)
    indices_centroids[1]=np.argmax(first_dist_arr)
    for i in range(2,k):
       #max_dist_arr = np.array([np.max(np.array([np.linalg.norm(x - centroid) for centroid in dataset[indices_centroids[0:i]]])) for x in dataset])
       max_dist_arr = np.array([np.mean(np.array([np.linalg.norm(x - centroid) for centroid in dataset[indices_centroids[0:i]]])) for x in dataset])
       indices_reversed_dist_arr= (np.argsort(max_dist_arr)[::-1])[0:k]
       indices_reversed_dist_arr=np.setdiff1d(indices_reversed_dist_arr, indices_centroids[0:i], assume_unique=True)
       indices_centroids[i]=indices_reversed_dist_arr[0]
    return dataset[indices_centroids]


def cluster_EM_step(coords_in, centroids_in, n_clusters):
    '''
    this function calls the function distance to compute the distance 
    between data points and clusters centroids.
    Then it assigns, according to the returned list each data point 
    to the cluster with the centroid which has shorted dist from the data point.
    Thein it estimates each new cluster centroid as the mean position between the 
    data points belonging to the cluster.
    At the end this function returns the new centroids and it continues like this
    '''
    distance_matrix = distance(coords_in, centroids_in,  n_clusters)
    distances_min = np.min(distance_matrix, axis=1)
    #estimate OBJECTIVE/COST function for k-means
    #cost-function := sum of the squared error -> sum of the squared Euclidean distances of each point to its closest centroid
    SSE_cost = np.sum(np.power(distances_min, 2))
    cluster_labels= np.argmin(distance_matrix, axis=1)
    new_centroids = np.zeros((n_clusters,2), dtype=np.float64)
    points_per_cluster= np.zeros(n_clusters, dtype=np.float64) 

    clusters = []
    for i_clust in range(0, n_clusters):
         arg_labels = np.where(cluster_labels== i_clust)
         cl_points = coords_in[arg_labels]
         clusters.append(cl_points)
         N_cluster_points = len(cl_points[:,0])
         points_per_cluster[i_clust] = N_cluster_points
         new_centroids[i_clust][0] = sum(cl_points[:,0])/N_cluster_points#mean x-coords of points in i-cluster
         new_centroids[i_clust][1] = sum(cl_points[:,1])/N_cluster_points#mean y-coord of points in i-cluster
    
    print('new centroids are:',new_centroids)
    return cluster_labels, points_per_cluster, SSE_cost, new_centroids




def K_means_algorithm(points_coords, K_sel, max_iter, tolerance, initialization_setup):
    if (initialization_setup=='plus'):
       centroids = guess_centroids_plus_plus(points_coords, K_sel)
    else:
       centroids = guess_centroids(points_coords, K_sel)
    
    print('initial centroids:', centroids)
    cost_arr = np.zeros(max_iter, dtype=np.float64)
    #MAIN LOOP
    iter = 0
    condition = True
    while condition==True and iter<max_iter:
        print("iteration: ",(iter+1))
        labels_arr, Npoints_per_cluster, SSE_cost, new_centroids = cluster_EM_step(points_coords, centroids, K_sel)
        print("average N_points per cluster: \n", np.mean(Npoints_per_cluster))
        print('objective function value is: ',SSE_cost)
        cost_arr[iter]= SSE_cost   
        if (np.linalg.norm(centroids- new_centroids)<tolerance):
           condition = False

        centroids= new_centroids
        iter=iter+1
    
    iterations_arr = np.arange(0,iter)
    return labels_arr, centroids, iter, cost_arr[iter-1]

def runs_for_K_means(Nruns, points_coords, K_sel, max_iteration, tolerance, initialization_setup):
	cost_arr= np.zeros(Nruns, dtype=np.float64)
	last_iter_arr= np.zeros(Nruns, dtype=np.float64)
	best_cost = 1.1e18 #large number to be sure that the last_cost condition is satisfied
	for i_run in range(Nruns):
		labels_arr, centroids, last_iter, last_cost = K_means_algorithm(points_coords, K_sel, max_iteration, tolerance, initialization_setup)
		cluster_data = pd.DataFrame( {'coord-0': points_coords[:,0], 'coord-1': points_coords[:,1], 'label': labels_arr, })
		group_by_cluster = cluster_data[['coord-0', 'coord-1', 'label']].groupby('label')
		counts_clusters = group_by_cluster.count()
		print(" N_points per cluster: \n", counts_clusters)
		cost_arr[i_run] = last_cost
		last_iter_arr[i_run]= last_iter
		if (i_run==0):
			best_cost= last_cost
			best_labels = labels_arr
			best_centroids = centroids
			#best_iter = last_iter
		else :
			if (last_cost < best_cost):
				best_cost = last_cost
				best_labels = labels_arr
				best_centroids = centroids
				#best_iter = last_iter
	return cost_arr, last_iter_arr, best_cost, best_labels, best_centroids
   
##################Reading INPUT DATA########################################################

#K_sel = int(input("Please enter the number of cluster you want:"))
option_dataset = str(input("Please enter the dataset option: A)Aggregation  or  B)s3"))
#dataset='S3'

if option_dataset=='A':
    dataset  ="Aggregation"
    dist_cut = 2.5
    delim = None
    K_sel = 7#6
    file_path = 'datasets/'
    filename = 'Aggregation.txt'
else:   
    dataset='S3'
    dist_cut = 57500
    delim = "    "
    K_sel = 15
    file_path = 'datasets/'
    filename = 's3.txt'

coord_data = read_input_file(file_path,filename, delimiter=delim)

points_coords = coord_data[:,0:2]



#max_iteration= int(input("please enter the maximum number of iterations that the algorithm should do."))
max_iteration = 100
tolerance=1.e-10
#tolerance=1.e-12


seed(100321)

#RUNNING 100 times the K-means algorithms 
#and returning the cluster identification associated with min loss function
cost_arr, last_iter_arr, best_cost, best_labels, best_centroids = runs_for_K_means(100, points_coords, K_sel, max_iteration, tolerance,  'standard')


#creation of dataframe for visualization
print('N_points', len(best_labels))
cluster_data = pd.DataFrame( {'coord-0': points_coords[:,0],
                              'coord-1': points_coords[:,1],
                              'label': best_labels,
                             })




#Clusters plot with K-means
best_run= np.argmin(cost_arr)
best_last_iter=last_iter_arr[best_run]
best_cost = cost_arr[best_run]
string_cost=str(best_cost)
string_liter= str(best_last_iter)

title_fig='(SSE value, last iter) at run '+ str(best_run) +'= ('+ string_cost +',' + string_liter + ')'
name_fig='plots/best_plot_clusters_for_K_'+str(K_sel)+'.png'
plt.figure(1, figsize=(15,10))
for i in range(K_sel):
    plt.scatter(cluster_data[cluster_data['label'] == i]['coord-0'] , cluster_data[cluster_data['label'] == i]['coord-1'] , label=i)
plt.scatter(best_centroids[:,0], best_centroids[:,1], s=18, marker="*")
plt.title(title_fig)
plt.legend(loc='best')
plt.savefig(name_fig)
plt.clf()

print('best cost value for K-means:', best_cost)
print('average cost for K-means:', np.mean(cost_arr))


#OUTPUT FILE with centroids coords
desired_fmt= '%8.2f', '%8.2f'
file_output_centroids="outputs/Kmeans_centroids_for_"+ dataset+"with_K_"+str(K_sel)+".txt"
file_output_centroids2="outputs/Kmeans_centroids_for_"+ dataset+"-with_increased_tolerance.txt"
data_output_centroids= np.column_stack((best_centroids[:,0], best_centroids[:,1]))
np.savetxt(file_output_centroids, data_output_centroids, fmt=desired_fmt, delimiter='  ', header='x-coord    y-coord')

seed(100321)
cost_arr_plus, last_iter_arr_plus, best_cost, best_labels_plus, best_centroids_plus = runs_for_K_means(100, points_coords, K_sel, max_iteration, tolerance,  'plus')



print('best cost value for K-means:', best_cost)
print('average cost for K-means:', np.mean(cost_arr_plus))

cluster_data['label-plus'] = best_labels_plus



best_run_plus= np.argmin(cost_arr_plus)
best_last_iter_plus=last_iter_arr_plus[best_run]
best_cost_plus = cost_arr_plus[best_run_plus]
string_cost=str(best_cost_plus)
string_liter= str(best_last_iter_plus)

#Plot with K-means++ 

title_fig='(SSE value, last iter) at run '+ str(best_run_plus) +'= ('+ string_cost +',' + string_liter + ')'
name_fig='plots/best_plot_clusters_for_K_'+str(K_sel)+'-Kplusplus.png'
name_fig2='plots/best_plot_clusters_for_'+dataset+'_K_'+str(K_sel)+'-Kplusplus_increased_tolerance.png'
plt.figure(1, figsize=(15,10))
plt.title(title_fig)
for i in range(K_sel):
    plt.scatter(cluster_data[cluster_data['label-plus'] == i]['coord-0'] , cluster_data[cluster_data['label-plus'] == i]['coord-1'] , label=i)
plt.scatter(best_centroids_plus[:,0], best_centroids_plus[:,1], s=35, marker="*", label='Kplus')
plt.scatter(best_centroids[:,0], best_centroids[:,1], s=45, marker="x", c='black', alpha=0.6, label='Kmeans')
plt.legend(loc='best')
plt.savefig(name_fig)
plt.show()


#OUTPUT FILE with cluster labels
dist_centers= distance(best_centroids_plus, best_centroids,K_sel)
best_labels_plus_resorted = itemgetter(best_labels_plus)(np.argmin(dist_centers, axis=1))

desired_fmt= '%8.2f', '%8.2f', '%10d', '%10d'
file_output2="outputs/Kmeans_labels_for_"+ dataset+"_increased_tolerance.txt"
file_output="outputs/Kmeans_labels_for_"+ dataset+"with_K_"+str(K_sel)+".txt"
data_output = np.column_stack((points_coords[:,0], points_coords[:,0], (best_labels+1), (best_labels_plus_resorted+1)))
np.savetxt(file_output, data_output, fmt=desired_fmt, delimiter='  ', header='x-coord    y-coord   Kmeans_labels  Kplus_labels ')


#SCREE PLOT (only for s3 dataset)

kk_arr= np.arange(2,21)
best_cost_arr=np.zeros(len(kk_arr), dtype=np.float64)

first_time='no'

if dataset=='S3' and first_time=='yes':
  for ik,kk in enumerate(kk_arr):
    cost_arr, last_iter_arr, best_cost, best_labels, best_centroids = runs_for_K_means(100, points_coords, kk, max_iteration, tolerance, 'standard')
    best_cost_arr[ik]= best_cost
    #saving cost_arr and iteration_arr for plotting later
    desired_fmt= '%8.2f', '%10d'
    file_name='outputs/Run_Ksel_'+str(kk)+'.txt'
    data_output = np.column_stack((cost_arr, last_iter_arr))
    np.savetxt(file_name,data_output,fmt=desired_fmt,delimiter='  ',header='Cost-value  iter_max')

  figname_Scree='plots/Scree_plot_Kmeans_S3.pdf'
  plt.figure(2)
  plt.xlabel('k')
  plt.ylabel('Cost function')
  plt.yscale('log')
  plt.xticks(kk_arr)
  plt.scatter(kk_arr, best_cost_arr)
  #plt.plot((iterations_arr+1), cost_arr[0:iter], '.')
  plt.savefig(figname_Scree)

else:
 if dataset=='S3':
   for ik,kk in enumerate(kk_arr):
    file_name='outputs/Run_Ksel_'+str(kk)+'.txt'
    data_kk=np.genfromtxt(file_name)
    best_cost_arr[ik] =np.min(data_kk[:,0])

   figname_Scree='plots/Scree_plot_Kmeans_S3.pdf'
   plt.figure(2)
   plt.xlabel('k')
   plt.ylabel('Cost function')
   plt.yscale('log')
   plt.xticks(kk_arr)
   plt.scatter(kk_arr, best_cost_arr)
   #plt.plot((iterations_arr+1), cost_arr[0:iter], '.')
   plt.savefig(figname_Scree)
































































