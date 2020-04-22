---
title: "T101: Embedded method-Feature selection techniques in machine learning"
date: 2020-03-11
categories: [Machine Learning]
tags: [Data Pre-processing,Feature Selection,Embedded Method,Lasso Regression,Ridge Regression]
excerpt: "A step by step guide on how to select features using embedded method"
toc: true
header:
  teaser: /assets/images/2020/03/11/Embedded_method.png
  image: /assets/images/2020/03/11/Embedded_method.png
  show_overlay_excerpt: False
---
**Note:** This is a part of series on Data Preprocessing in Machine Learning you can check all tutorials here: [Embedded Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-embedded-method/), [Wrapper Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-wrapper-method/), [Filter Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-filter_method/),[Handling Multicollinearity](https://datamould.github.io/machine%20learning/2020/03/13/feature-selection-handling-multicollinearity/).
{: .notice--info}

Are you sinking into lots of feature but do not know which one to pick and which one to ignore?
Then how would you develop a predictive model?

![gif](/assets/images/2020/03/11/feature_selection.gif)

This is one of those questions which every machine learning engineer comes accross, you need deep knowledge of that domain to give an accepted answer,but don't worry I am going to help you automating this process in this tutorial, there are certain checklists we should follow to select the best features from our data.
Let's begin!!

## What is Feature Selection?

Feature selection is the automated process of selecting important features out of all the features in our dataset.

## Why we need it?
{:.no_toc}
Feature selection helps the model to increase its accuracy and improve the computational efficiency.

## Feature selection vs Dimensionality reduction?

Feature selection isn't like dimensionality reduction. Both methods are used to lessen the quantity of features/attributes in the dataset, however a dimensionality reduction technique accomplish that by way of developing new combos of features, where as feature selection techniques include and exclude features present within the dataset without changing them.

* Dimensionality reduction techniques : Principal Component Analysis, Singular Value Decomposition.
* Feature Selection techniques : Filter method, Wrapper method, Embedded method



```python
import pandas as pd
import numpy as np
```


```python
automobile=pd.read_csv('dataset/cleaned_cars.csv')
automobile.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>origin</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>US</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>US</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>US</td>
      <td>49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>US</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.0</td>
      <td>8</td>
      <td>429.0</td>
      <td>198</td>
      <td>4341</td>
      <td>10.0</td>
      <td>US</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
</div>



Here we are using a car's dataset, we are dropping the target column "mpg".
In this tutorial our primary goal is to learn feature selection techniques hence we are dropping "origin" column as well as it is not a continuous variable.


```python
X=automobile.drop(['mpg','origin'],axis=1)
Y=automobile['mpg']
X,Y
```




    (     cylinders  displacement  horsepower  weight  acceleration  Age
     0            8         307.0         130    3504          12.0   49
     1            8         350.0         165    3693          11.5   49
     2            8         318.0         150    3436          11.0   49
     3            8         302.0         140    3449          10.5   49
     4            8         429.0         198    4341          10.0   49
     ..         ...           ...         ...     ...           ...  ...
     362          4         151.0          90    2950          17.3   37
     363          4         140.0          86    2790          15.6   37
     364          4          97.0          52    2130          24.6   37
     365          4         135.0          84    2295          11.6   37
     366          4         120.0          79    2625          18.6   37
     
     [367 rows x 6 columns], 0      18.0
     1      15.0
     2      18.0
     3      17.0
     4      15.0
            ... 
     362    27.0
     363    27.0
     364    44.0
     365    32.0
     366    28.0
     Name: mpg, Length: 367, dtype: float64)



## Embedded Method
Embedded methods selects the important features while the model is being trained, You can say few model training algorithms already implements a feature selection process while getting trained with the data.

In this example we will be discussing about Lasso Regression , Ridge regression , decision tree.

### Lasso Regression

Lasso regression is a L1 regularized regression when there is a penalty for more complicated coefficients.


```python
from sklearn.linear_model import Lasso
```


```python
lasso=Lasso(alpha=.8)
lasso.fit(X,Y)
```




    Lasso(alpha=0.8, copy_X=True, fit_intercept=True, max_iter=1000,
          normalize=False, positive=False, precompute=False, random_state=None,
          selection='cyclic', tol=0.0001, warm_start=False)



Alpha value determines the strength of the regularization of our model , penalty parameter is the sum of the abs value of the coefficients and penalty parameter is multiplied by the alpha parameter.

Once we fit the lasso regression there is one property of coefficient for each features.
Regularization parameters in lasso regression forces unimportant parameters coefficients close to 0.


```python
predictors=X.columns
coef=pd.Series(lasso.coef_,predictors).sort_values()
print(coef)
```

    Age            -0.666233
    horsepower     -0.008517
    weight         -0.006472
    displacement   -0.000602
    cylinders      -0.000000
    acceleration    0.000000
    dtype: float64
    

So we can see that Age and Weight are the most significant features among these predictors and these features are called lasso features.


```python
lasso_features=['Age','weight']
lasso_feature_df=X[lasso_features]
lasso_feature_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>3504</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>3693</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>3436</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49</td>
      <td>3449</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49</td>
      <td>4341</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>362</th>
      <td>37</td>
      <td>2950</td>
    </tr>
    <tr>
      <th>363</th>
      <td>37</td>
      <td>2790</td>
    </tr>
    <tr>
      <th>364</th>
      <td>37</td>
      <td>2130</td>
    </tr>
    <tr>
      <th>365</th>
      <td>37</td>
      <td>2295</td>
    </tr>
    <tr>
      <th>366</th>
      <td>37</td>
      <td>2625</td>
    </tr>
  </tbody>
</table>
<p>367 rows × 2 columns</p>
</div>



### Ridge Regression

Ridge regression is a L2 regularization technique.


```python
from sklearn.linear_model import Ridge
```


```python
ridge=Ridge(alpha=1.0)
ridge.fit(X,Y)
```




    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)



Alpha value determines the strength of the regularization of our model , penalty parameter is the sum of the square of abs value of the coefficients and penalty parameter is multiplied by the alpha parameter.

Once we fit the Ridge regression there is one property of coefficient for each features.
Regularization parameters in Ridge regression forces unimportant parameters coefficients close to 0, for correlated features, it means that they tend to get similar coefficients.Feature having negative coefficients don't contribute that much to the model.


```python
predictors=X.columns
coef=pd.Series(ridge.coef_,predictors).sort_values()
print(coef)
```

    Age            -0.738056
    cylinders      -0.174340
    weight         -0.006706
    horsepower     -0.001467
    displacement    0.003505
    acceleration    0.072909
    dtype: float64
    


```python
ridge_features=['displacement','acceleration']
ridge_feature_df=X[ridge_features]
ridge_feature_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>displacement</th>
      <th>acceleration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>307.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>350.0</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>318.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>302.0</td>
      <td>10.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>429.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>362</th>
      <td>151.0</td>
      <td>17.3</td>
    </tr>
    <tr>
      <th>363</th>
      <td>140.0</td>
      <td>15.6</td>
    </tr>
    <tr>
      <th>364</th>
      <td>97.0</td>
      <td>24.6</td>
    </tr>
    <tr>
      <th>365</th>
      <td>135.0</td>
      <td>11.6</td>
    </tr>
    <tr>
      <th>366</th>
      <td>120.0</td>
      <td>18.6</td>
    </tr>
  </tbody>
</table>
<p>367 rows × 2 columns</p>
</div>



### Decision Tree

During the construction of a decision tree the structure of the decision tree is such that the more important features are higher up , are closer to the root.



```python
from sklearn.tree import DecisionTreeRegressor
```


```python
decision_tree= DecisionTreeRegressor(max_depth=4)
decision_tree.fit(X,Y)
```




    DecisionTreeRegressor(criterion='mse', max_depth=4, max_features=None,
                          max_leaf_nodes=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          presort=False, random_state=None, splitter='best')




```python
predictors=X.columns
coef=pd.Series(decision_tree.feature_importances_,predictors).sort_values()
print(coef)
```

    cylinders       0.000000
    acceleration    0.003211
    weight          0.048203
    Age             0.103696
    horsepower      0.188937
    displacement    0.655953
    dtype: float64
    

So we can see the most significant features are displacement and horsepower


```python
decision_tree_features=['displacement','horsepower']
```


```python
decision_tree_feature_df=X[decision_tree_features]
```


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
def buildmodel(X,Y,test_frac,model_name=''):
    x_train,x_test,y_train,y_test =train_test_split(X,Y,test_size=test_frac)
    model =LinearRegression().fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print('Accuracy of the '+model_name+' model is '+str(r2_score(y_test,y_pred)))
```

Now let's check the accuracy of these models


```python
buildmodel(ridge_feature_df,Y,test_frac=.2,model_name='Ridge')
```

    Accuracy of the Ridge model is 0.6524069499984987
    


```python
buildmodel(lasso_feature_df,Y,test_frac=.2,model_name='Lasso')
```

    Accuracy of the Lasso model is 0.823518232271492
    


```python
buildmodel(decision_tree_feature_df,Y,test_frac=.2, model_name='Decision Tree')
```

    Accuracy of the Decision Tree model is 0.5284130942943683

You can get the notebook used in this tutorial [here](https://github.com/arupbhunia/Data-Pre-processing/blob/master/FeatureSelectionUsingEmbeddedMethod.ipynb) and dataset used [here](https://github.com/arupbhunia/Data-Pre-processing/blob/master/datasets/datasets/cleaned_cars.csv)
    
Thanks for reading!!