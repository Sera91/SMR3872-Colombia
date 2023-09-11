import joblib
import time
import pandas as pd
#import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
       # Dataset for testing purposes
       X, y = make_classification(n_classes=2, n_features=6, n_samples=500, n_informative=2, scale=100, random_state=12)   
           
       #conversion from ndarrays to dataframes
       df = pd.DataFrame(X, columns=['var'+str(i) for i in range(1, X.shape[1]+1)])
       df['label'] = y

       #Train test split
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

       steps = [('scaler', StandardScaler()),
         ('svm_classif', SVC(kernel='rbf', gamma=0.5, C=10))]# Create Pipeline object
       rbf_kernel = Pipeline(steps)# Run the pipeline (fit) 
       #Scale data and Fit the model
       rbf_kernel.fit(X_train,y_train)# Predictions
       preds = rbf_kernel.predict(X_test)
       # performance dataframe
       result = pd.DataFrame(X_test, columns=['var'+str(i) for i in range(1, X.shape[1]+1)])
       result['preds'] = preds# Plot var1(on x) and var5(on y)
       plt.figure(1)
       plt.scatter(x=results['var1'], y=results['var5'])
       plt.legend()
       plt.savefig("scatterplot-predictions.png")

       #using seaborn this plot becomes one line of code:

       #sns.scatterplot(data=result, x='var1', y='var5', hue='preds')


       # Confusion Matrix
       conf_matrix_SVC = pd.DataFrame(confusion_matrix(y_test, result.preds, labels=[0,1]))
       print(conf_matrix_SVC)

       steps_RF = [('scaler', StandardScaler()), ('rf_classif', RandomForestClassifier())]# Create pipeline
       rf = Pipeline(steps_RF)# Fit
       rf.fit(X_train,y_train)# Preds
       preds = rf.predict(X_test)# performance
       result = pd.DataFrame(X_test, columns=['var'+str(i) for i in range(1, X.shape[1]+1)])
       result['preds'] = preds# Confusion Matrix
       conf_matrix_RF = pd.DataFrame(confusion_matrix(y_test, result.preds, labels=[0,1]))
       print(conf_matrix_RF)

       
