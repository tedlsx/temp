---
title: Machine Learning 
---
Before we use many ML algorithm, we sometimes need to preprocess the data

Import all package
```python
import numpy as np
import pandas as pd
import statsmodels as sm

from sklearn import preprocessing
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#import model package
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
```

## Preprocessing
Function to encode the data type, change all object type features into dummy variables 
```python 
def encoder(df):
    for column in df.columns:
        if df[column].dtype == type(object):
            le = preprocessing.LabelEncoder()
            df[column] = le.fit_transform(df[column])
    return df
```
Function to normalise the data
```python
def normalised(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df.values)
    df_new = pd.DataFrame(x_scaled, columns= df.columns)
    return df_new
```

Or we can use the package in scikit learn of [data transformation](https://scikit-learn.org/stable/data_transforms.html). Where fit() is to calculate properties such like mean, min, max and so on for training process.  transform() is to apply normalise, reduce dimension and regularization base on fitted data. fit_transform combined these two methods.
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_tranform(x_train)
sc.tranform(x_test)

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit_tranform(x_train)
mms.tranform(x_test)
```

Using PCA as an example of reducing dimension
```python
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca.fit_transdorm()
```



Spliting the data frame into train and test to varify the goodness of model
```python
# a fixed randome_state num will have same train and test set 
train_x, test_x, train_y, test_y = train_test_split(dfhouse, df_encode[["prod_id"]], test_size=0.2, random_state=42)
```
Logistic Regression Classification
```python
def logistic_regression_classifier(train_x, train_y):
    model = LogisticRegression(penalty='l2', solver = "lbfgs", multi_class='auto')
    model.fit(train_x, train_y)
    return model

lgr_model = logistic_regression_classifier(train_x, train_y)
pred_y = lgr_model.predict(test_x)
metrics.accuracy_score(test_y, pred_y)
```

Naive Bayes Classification
```python
def naive_bayes_classifier(x, y):
    model = GaussianNB()
    model.fit(x, y)
    return model

nb_model = naive_bayes_classifier(train_x, train_y)
pred_y = nb_model.predict(test_x)
metrics.accuracy_score(test_y, pred_y)
```

KNN K-Nearest Neighbors Classification
```python
def knn_classifier(x, y):
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model 

knn_model = knn_classifier(train_x, train_y)
pred_y = knn_model.predict(test_x)
metrics.accuracy_score(test_y, pred_y)
```

Random Forest Classifier
```python
def random_forest_classifier(train_x, train_y):
    model = RandomForestClassifier(max_depth=2, random_state=0, )
    model.fit(train_x, train_y)
    return model

rf_model = random_forest_classifier(train_x, train_y)
rf_model.feature_importances_  # show feature importance for each feature
model = SelectFromModel(rf_model, prefit=True) # select no zero coefficient features
x_new = model.transform(x)
x_new.shape()

train_x_rf = train_x.iloc[:, my_list] # can mutually define my_list = [] to select important feature
metrics.accuracy_score(test_y, pred_y)
```

Decision Tree Classifier
```python
def decision_tree_classifier(train_x, train_y):
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model

dt_model = decision_tree_classifier(train_x, train_y)
pred_y = dt_model.predict(test_x)
metrics.accuracy_score(test_y, pred_y)
```

GBDT-Gradient Boosting Decision Tree Classification
```python
def gradient_boosting_classifier(train_x, train_y):
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model

gbdt_model = gradient_boosting_classifier(train_x, train_y)
pred_y = gbdt_model.predict(test_x)
metrics.accuracy_score(test_y, pred_y)
```

SVM-Support Vaector Machine Classification
```python
def svm_classifier(train_x, train_y):
    model = SVC(kernel='rbf', probability=True, gamma = "auto")
    model.fit(train_x, train_y)
    return model

svm_model = svm_classifier(train_x, train_y)
pred_y = svm_model.predict(test_x)
metrics.accuracy_score(test_y, pred_y)
```