#Fuzzy-c-means Algorithm
import os
from math import sqrt
import random
from random import seed
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fcmeans import FCM
from joblib import Parallel, delayed, cpu_count

m= 2.0 #fuzzification parameter




pd.options.display.max_rows = 4000

def read_input_file(path_to_file, filename, delimiter=","):
    input_data = np.genfromtxt(os.path.join(path_to_file, filename), delimiter=delimiter, dtype=float)
    print(input_data[0:3])
    print(len(input_data[:,0]))
    return input_data


def distance(data_points, centroids, N_clusters):
    N_points = len(data_points[:,0])
    #print(N_points)
    dist_matrix =np.zeros((N_points, N_clusters), dtype=float)
    #distances_arr = []
    for i, centroid in enumerate(centroids):
            dist_matrix[:,i] = np.sqrt((data_points[:,0] -centroid[0])**2 + (data_points[:,1]-centroid[1])**2)
    return dist_matrix



def distance_mg(data_points, centroids, N_clusters):
    mgx = np.meshgrid(data_points[:,0], centroids[:,0])
    mgy = np.meshgrid(data_points[:,1], centroids[:,1])
    N_points = len(data_points[:,0])
    return np.sqrt((mgx[0].ravel() - mgx[1].ravel())**2 + (mgy[0].ravel() - mgy[1].ravel())**2).reshape(N_clusters,N_points).T



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


def compute_Loss_fcmeans(data, w_probs, centroids, m):
    squared_distance = np.sum(np.square(np.tile(centroids,(data.shape[0],1,1)) - data.reshape((data.shape[0],1,data.shape[1]))), axis=2)
    power_probs = np.power(w_probs, m)
    return np.sum(np.multiply(squared_distance, power_probs))

def assign_init_probs(N_points, n_clusters):
    raw_mat = np.random.uniform(0.0,1.0,size=(N_points, n_clusters))
    out = np.divide(raw_mat, np.sum(raw_mat, axis=1).reshape((N_points, 1)))
    return out


def pair_dist(point_A,point_B):
    return np.linalg.norm(point_A - point_B)

def arg_U_k(point_A, centroid_A, exp):
    return pow(np.linalg.norm(point_A - centroid_A), exp)


def update_W_prob(N_points, points, n_clusters, centroids, exp): 
    W_matrix = np.zeros((N_points, n_clusters), dtype=np.float64)
    for i in range(N_points):
      arr_denom = np.power(np.array([np.linalg.norm(points[i] - centroid) for centroid in centroids]), exp)
      W_matrix[i] = np.array([1.0/np.sum(pow(np.linalg.norm(points[i] - centroid_i), exp)/arr_denom) for centroid_i in centroids])
    return W_matrix



def update_W_prob_vect(N_points, points, n_clusters, centroids, exp):
    d_matrix= distance(points, centroids, n_clusters)
    d_matrix_exp = np.power(d_matrix, exp)
    d_col= np.sum(1.0/d_matrix_exp, axis=1)
    return 1.0/ (d_matrix_exp*d_col[:,np.newaxis])
    


def fuzzy_cluster_EM_step(coords_in, centroids_in, n_points, n_clusters, W_matrix, m):
    '''
    this function calls the function distance to compute the distance 
    between data points and clusters centroids.
    Then it assigns, according to the returned list each data point 
    to the cluster with the centroid which has shorted dist from the data point.
    Thein it estimates each new cluster centroid as the mean position between the 
    data points belonging to the cluster.
    At the end this function returns the new centroids and it continues like this
    '''
    #distance_matrix = distance(coords_in, centroids_in,  n_clusters)
   
    exp= (2.0/(m-1))

    power_matrix = np.power(W_matrix, m)
    denominators = np.sum(power_matrix, axis=0)
   
    new_cl_centroids = np.array([np.array([np.sum(power_matrix[:,i]*coords_in[:,0]), np.sum(power_matrix[:,i]*coords_in[:,1])])/denominators[i] for i in range(n_clusters)])

    #new_W_matrix= update_W_prob(n_points, coords_in, n_clusters, new_cl_centroids, exp)
    new_W_matrix= update_W_prob_vect(n_points, coords_in, n_clusters, new_cl_centroids, exp)
    #print('W_matrix first-row:', np.max(W_matrix[0]))
    
    #estimate OBJECTIVE/COST function for fuzzy c-means
    #cost-function := sum of the squared error -> sum of the squared Euclidean distances of each point to its closest centroid
    SSE_cost = compute_Loss_fcmeans(coords_in, new_W_matrix, new_cl_centroids, m)
    #print('new centroids are:',new_cl_centroids)
    return SSE_cost, new_cl_centroids, new_W_matrix




def fcmeans_algo(points_coords, n_clusters, max_iteration, tolerance, initialization_setup):
    if (initialization_setup=='plus'):
       centroids = guess_centroids_plus_plus(points_coords, n_clusters) +[1,1]
    else:
       centroids = guess_centroids(points_coords, n_clusters) + [1,1]

    n_points = points_coords.shape[0]
    print('initial centroids:', centroids)
    #centorids= kmeanspp_init(points_coords, n_clusters) +[1,1]            # init clusters with k-means++
    # the +[1,1] is needed, otherwise in the first update of the
    # assign_probs the function fails due to zero division err 
    W_probs = assign_init_probs(points_coords.shape[0], n_clusters) # init assign_probs randomly
    #print('max iteration:', max_iteration)
    rep=0
    changed = True
    while(changed and rep<max_iteration):
        SSE_cost, new_centroids, new_W_matrix = fuzzy_cluster_EM_step(points_coords, centroids, n_points, n_clusters, W_probs, 2)
        changed = True if np.linalg.norm(new_centroids - centroids)> tolerance else False
        centroids = new_centroids
        W_probs   = new_W_matrix
        rep = rep+ 1
    print('new centroids are:',centroids)
    print('last iteration:', rep)
    return rep, SSE_cost, centroids, W_probs


def runs_for_fuzzyCmeans(points_coords, K_sel, max_iteration, tolerance, initialization_setup, Nruns=100):
        
        _results = Parallel(n_jobs=-1)(delayed(fcmeans_algo)(points_coords, K_sel, max_iteration, tolerance, initialization_setup) for i in range(Nruns))
        _last_iter_arr = [_results[i][0] for i in range(Nruns)]
        _cost_arr = [_results[i][1] for i in range(Nruns)]
        _best_results = _results[np.argmin(_cost_arr)]
        return _cost_arr, _last_iter_arr, _best_results[1], _best_results[2], _best_results[3]

	#cost_arr= np.zeros(Nruns, dtype=np.float64)
	#last_iter_arr= np.zeros(Nruns, dtype=np.float64)
	#best_cost = 1.1e18 #large number to be sure that the last_cost condition is satisfied
	#for i_run in range(Nruns):
	#	last_iter, last_cost, centroids, W_probs = fcmeans_algo(points_coords, K_sel, max_iteration, tolerance, initialization_setup)
	#	cost_arr[i_run] = last_cost
	#	last_iter_arr[i_run]= last_iter
	
       	#	if (last_cost < best_cost):
	#			best_cost      = last_cost
	#			best_W_probs   = W_probs
	#			best_centroids = centroids
	#			#best_iter = last_iter
	#return cost_arr, last_iter_arr, best_cost, best_centroids, best_W_probs



#MAIN 

#Reading INPUT DATA

option_dataset = str(input("Please enter the dataset option: A) Aggregation  or  B) s3"))

if  option =='A':
    dataset ="Aggregation"
    delim = None
    file_path = 'datasets/'
    filename = 'Aggregation.txt'
    k_sel= 7
else:   #S3 dataset
    dataset='S3'
    delim = "    "
    file_path = 'datasets/'
    filename = 's3.txt'
    k_sel = 15

coord_data = read_input_file(file_path,filename, delimiter=delim)

points_coords = coord_data[:,0:2]


#asking for requested number of cluster

#k_sel = int(input("Please enter the number of cluster you want:"))



max_iter = 100
print('max iteration:', max_iter)
tolerance=1.e-8


seed(100321)

#last_iter, last_cost, centroids, W_probs = fcmeans_algo(points_coords, K_sel, max_iter, tolerance, 'standard')


cost_arr, last_iter_arr, best_cost, best_centroids, best_W_probs = runs_for_fuzzyCmeans(points_coords, k_sel, max_iter, tolerance,  'standard')

avg_cost = np.mean(cost_arr); 



best_labels= np.argmax(best_W_probs, axis=1)
print(best_labels.size)

#Clusters plot
best_run= np.argmin(cost_arr)
best_last_iter=last_iter_arr[best_run]
best_cost_K= cost_arr[best_run]
print("test diff best-loss: ", (best_cost_K-best_cost))
string_cost=str(best_cost)
string_liter= str(best_last_iter)
string_best_run=str(best_run)


#best_centroids=centroids
#best_labels= np.argmax(W_probs, axis=1)
#string_cost=str(last_cost)
#string_liter= str(last_iter)
#string_best_run=str(last_iter)

sorted_best_centroids = best_centroids[np.argsort(best_centroids[:,0])]

print(best_centroids)


#creation of dataframe for visualization
print('N_points', len(best_labels))
cluster_data = pd.DataFrame( {'coord-0': points_coords[:,0],
                              'coord-1': points_coords[:,1],
                              'label': best_labels,
                             })


title_fig='(SSE value, last iter) at run '+ string_best_run +'= ('+ string_cost +',' + string_liter + ')'
name_fig='plots/best_plot_clusters_for_Fuzzy_C_means_for'+dataset+'_K_'+str(k_sel)+'_new.png'
plt.figure(1, figsize=(15,10))
for i in range(k_sel):
    plt.scatter(cluster_data[cluster_data['label'] == i]['coord-0'] , cluster_data[cluster_data['label'] == i]['coord-1'] , label=i, alpha=0.2)
plt.scatter(best_centroids[:,0], best_centroids[:,1], s=18, marker="*")
plt.title(title_fig)
plt.legend(loc='best')
plt.savefig(name_fig)
plt.show()
plt.clf()

#comparing with a library for fuzzy clustering 
X=coord_data[:,0:2]

fcm = FCM(n_clusters=k_sel)
fcm.fit(X)


# outputs
fcm_centers = fcm.centers
fcm_labels = fcm.predict(X)


sorted_fcm_centers = fcm_centers[np.argsort(fcm_centers[:,0])]



print("comparing predictions of my routine with fuzzy-c-means library:")
print("diff between sorted centroids", (sorted_fcm_centers - sorted_best_centroids))


#print("shape labels:", fcm_labels.shape)

# plot result
f, axes = plt.subplots(1, 2, figsize=(11,5))
axes[0].scatter(cluster_data['coord-0'] , cluster_data['coord-1'] , c=best_labels, alpha=.1)
axes[0].scatter(best_centroids[:,0], best_centroids[:,1], marker="+", s=500, c='w')
axes[1].scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
plt.savefig('plots/comparing-fuzzy-clustering-'+dataset+'.jpg')
plt.clf()
#plt.show()



#SCREE PLOT

kk_arr= np.arange(3,21)
best_cost_arr=np.zeros(len(kk_arr), dtype=np.float64)

first_time='no'

if first_time=='yes':
 for ik,kk in enumerate(kk_arr):
    cost_arr, last_iter_arr, best_cost, best_centroids, best_W_probs = runs_for_fuzzyCmeans(points_coords, kk, max_iter, tolerance, 'standard')
    best_cost_arr[ik]= best_cost
    #saving cost_arr and iteration_arr for plotting later
    desired_fmt= '%8.2f', '%10d'
    file_name='outputs/Run_Ksel_'+str(kk)+'_fuzzy.txt'
    data_output = np.column_stack((cost_arr, last_iter_arr))
    np.savetxt(file_name,data_output,fmt=desired_fmt,delimiter='  ',header='Cost-value  iter_max')

else:
  for ik,kk in enumerate(kk_arr):
    file_name='outputs/Run_Ksel_'+str(kk)+'_fuzzy.txt'
    data_kk=np.genfromtxt(file_name)
    best_cost_arr[ik] =np.min(data_kk[:,0])




figname_Scree='plots/Scree_plot_FuzzyCmeans.pdf'
plt.figure(2)
plt.xlabel('k')
plt.ylabel('Cost function')
plt.yscale('log')
plt.xticks(kk_arr)
plt.scatter(kk_arr, best_cost_arr)
#plt.plot((iterations_arr+1), cost_arr[0:iter], '.')
plt.savefig(figname_Scree)


