---
title: Find important feature
---
## Select feature
### Reason to do feature selection
1. Because a large number of features may cost long training time
2. Increasing number of features may increase the risk of overfitting
It can also help us to reduce the dimension of our dataset without loss main information.
### Methods of feature selection
#### P-value
P-value is the probability of Null hypothesis is true in statisitc model. Normally we select p-value = 0.05 as an significant level.
```python
import statsmodels.api as sm
#Adding constant column of ones, mandatory for sm.OLS model
x = pd.DataFrame() # feature df
y = pd.DataFrame() # target df
x_1 = sm.add_constant(x) # add a constant column to df x
#Fitting sm.OLS model
model = sm.OLS(np.array(y), np.array(x_1)).fit()
# Which prints out the p-value of each features in this model as an array 
model.pvalues
```

Write a function to select feature based on p-value:
```python
# x is feature df
# y is target df
# sl is significant level
def backwardElimination(x, y, sl):
    cols = list(x.columns)
    pmax = 1
    while (len(cols) > 0):
        p= []                                                                                   
        X_1 = x[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index = cols)      
        pmax = max(p)
        # .idxmax returns the index of maximum
        feature_with_p_max = p.idxmax()
        if(pmax > sl):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    print(selected_features_BE)

backwardElimination(x, y, 0.05)
```


#### F-test
F-test is a statistical test to find whether there is a significant difference between two model. Least square error is calculated for each model and compared.

Here introduced the [skitlearn package](https://scikit-learn.org/stable/), we will use [F-test](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html) to find first K best features:

For continues data type
```python
from sklearn import sklearn.feature_selection
sklearn.feature_selection.f_regression(x, y) # where x is feature df(n_sample * n_features), y is target df (n_samples)
# output is set of F-score and p-value for each F-score
```
For classification data
```python
sklearn.feature_selection.f_classif(x, y) # same with f_regression
```
F-score is good for linear relation

#### Mutual infomation
Which is good for non-linear relation
```python
sklearn.feature_selection.mututal_info_regression 
sklearn.feature_selection.mututal_info_classif
```
