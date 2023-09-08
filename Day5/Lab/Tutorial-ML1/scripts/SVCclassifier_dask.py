from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from dask_mpi import initialize
from dask.distributed import Client, as_completed
import dask.dataframe as dd
import joblib
import time


#The digits dataset is a dataset 


if __name__ == "__main__":
         

       print('I am before  client inizialization')
       initialize(memory_limit=2.e9)

       dask_client = Client()

       dask_client.wait_for_workers(n_workers=8)
       #dask_client.restart()

       num_workers = len(dask_client.scheduler_info()['workers'])
       print("%d workers available and ready"%num_workers)



       X, y = load_digits(return_X_y=True)

       

       print(type(X))
       print(X.shape) #shape of data: 64 cols with 1797 obs
       print(X[:,0].size)


       param_grid = {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
              "kernel": ['rbf', 'poly', 'sigmoid'],
              "shrinking": [True, False]}

       grid_search = GridSearchCV(SVC(gamma='auto', random_state=0, probability=True),
                           param_grid=param_grid,
                           return_train_score=False,
                           cv=3,
                           n_jobs=-1)

       t_start = time.time()
       with joblib.parallel_backend('dask'):
            grid_search.fit(X, y)
       t_end = time.time()
       Delta_t= t_end - t_start
       print("Delta time to fit and cross-validate SVC model with Dask :", Delta_t)
       print('computation finished')
