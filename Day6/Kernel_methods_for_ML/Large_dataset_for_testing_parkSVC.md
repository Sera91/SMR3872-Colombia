#Instructions to preprocess the ADULT input dataset
First, download the dataset from the web:
wget https://archive.ics.uci.edu/static/public/2/adult.zip

1)Read the dataset from the csv file using pandas:

data = pd.read_csv('file.csv')

list_features= data.columns.to_list()

categorica_features = data.select_dtypes(include=['object']).columns.tolist()

numerical_features = data.select_dtypes(include=['int','float']).columns.tolist()



2)Then encode the categorical variables into integer values. 
You have two options.

Option A:


one_hot_encoded_data = pd.get_dummies(data, columns = categorical_features)
print(one_hot_encoded_data)

Option B:
from sklearn.preprocessing import OneHotEncoder

#for example: categoric-feature= workclass

# Create an instance of One-hot-encoder
enc = OneHotEncoder()
  
# Passing encoded columns
  
enc_data = pd.DataFrame(enc.fit_transform(
    data[categorical_features]).toarray())

data = data.join()


3) Rescale the numerical feature usinf Standard Scaler
from sklearn.preprocessing import StandardScaler

# example: numeric_features = "Age"

scale = StandardScaler()
  
# Passing encoded columns
  
rescaled_data = pd.DataFrame(scale.fit_transform(
    data[numerical_features]).toarray())

data = data.join()




ALTERNATIVE) You can do step 2 and 3 together as:
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ["",""]

numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler()),]
)

categorical_features = ["",""]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


STEP 4) Put preprocessor and classifier together:

from sklearn import svm

SVC = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo')

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", SVC)]
)

STEP 5) fitting the model to the data

y = data["income"]

X_data = data.drop(['income', ], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=0)

clf.fit(X_train, y_train)


