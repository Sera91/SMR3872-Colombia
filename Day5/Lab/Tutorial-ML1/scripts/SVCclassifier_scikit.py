from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import joblib
import time


#The digits dataset is a dataset 


if __name__ == "__main__":
  
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
       grid_search.fit(X, y)
       t_end = time.time()

       Delta_t= t_end - t_start
       print("Delta time to fit and cross-validate SVC model with scipy:", Delta_t)

       
