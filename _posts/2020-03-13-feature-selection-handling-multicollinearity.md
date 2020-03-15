---
title: "T104: Handling Multicollinearity-Feature selection techniques in machine learning"
date: 2020-03-13
categories: [Machine Learning]
tags: [Data Pre-processing,Feature Selection,Handling Multicollinearity,VIF]
excerpt: "A step by step guide on how to select features by handling Multi collinearity"
toc: true
header:
  teaser: /assets/images/2020/03/13/handling_multicollinearity.png
  image: /assets/images/2020/03/13/handling_multicollinearity.png
  show_overlay_excerpt: False
mathjax: true  
---

**Note:** This is a part of series on Data Preprocessing in Machine Learning you can check all tutorials here: [Embedded Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-embedded-method/), [Wrapper Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-wrapper-method/), [Filter Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-filter-method/),[Handling Multicollinearity](https://datamould.github.io/machine%20learning/2020/03/13/feature-selection-handling-multicollinearity/).
{: .notice--info}

In this tutorial we will learn how to handle multicollinear features , this can be performed as a feature selection step in your machine learning pipeline.
When two or more independent variables are highly correlated with each other then we can state that those features are multi collinear.


```python
import pandas as pd
import numpy as np

```


```python
cars_df=pd.read_csv('dataset/cleaned_cars.csv')
cars_df.head()
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




```python
cars_df.describe()
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
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>367.000000</td>
      <td>367.000000</td>
      <td>367.000000</td>
      <td>367.000000</td>
      <td>367.000000</td>
      <td>367.000000</td>
      <td>367.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.556403</td>
      <td>5.438692</td>
      <td>191.592643</td>
      <td>103.618529</td>
      <td>2955.242507</td>
      <td>15.543324</td>
      <td>42.953678</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.773266</td>
      <td>1.694068</td>
      <td>102.017066</td>
      <td>37.381309</td>
      <td>831.031730</td>
      <td>2.728949</td>
      <td>3.698402</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>70.000000</td>
      <td>46.000000</td>
      <td>1613.000000</td>
      <td>8.000000</td>
      <td>37.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.550000</td>
      <td>4.000000</td>
      <td>105.000000</td>
      <td>75.000000</td>
      <td>2229.000000</td>
      <td>13.800000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23.000000</td>
      <td>4.000000</td>
      <td>146.000000</td>
      <td>94.000000</td>
      <td>2789.000000</td>
      <td>15.500000</td>
      <td>43.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>29.000000</td>
      <td>8.000000</td>
      <td>260.000000</td>
      <td>121.000000</td>
      <td>3572.000000</td>
      <td>17.050000</td>
      <td>46.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>46.600000</td>
      <td>8.000000</td>
      <td>455.000000</td>
      <td>230.000000</td>
      <td>5140.000000</td>
      <td>24.800000</td>
      <td>49.000000</td>
    </tr>
  </tbody>
</table>
</div>



As we can see range of these features are very different that means they all are in different scales so lets standardize the features using sklearn's scale function.


```python
from sklearn import preprocessing

cars_df[['cylinders']]=preprocessing.scale(cars_df[['cylinders']].astype('float64'))
cars_df[['displacement']]=preprocessing.scale(cars_df[['displacement']].astype('float64'))
cars_df[['horsepower']]=preprocessing.scale(cars_df[['horsepower']].astype('float64'))
cars_df[['weight']]=preprocessing.scale(cars_df[['weight']].astype('float64'))
cars_df[['acceleration']]=preprocessing.scale(cars_df[['acceleration']].astype('float64'))
cars_df[['Age']]=preprocessing.scale(cars_df[['Age']].astype('float64'))

```


```python
cars_df.describe()
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
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>367.000000</td>
      <td>3.670000e+02</td>
      <td>3.670000e+02</td>
      <td>3.670000e+02</td>
      <td>3.670000e+02</td>
      <td>3.670000e+02</td>
      <td>3.670000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.556403</td>
      <td>-1.936084e-17</td>
      <td>-1.936084e-17</td>
      <td>9.680419e-17</td>
      <td>-7.744335e-17</td>
      <td>9.680419e-17</td>
      <td>2.323300e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.773266</td>
      <td>1.001365e+00</td>
      <td>1.001365e+00</td>
      <td>1.001365e+00</td>
      <td>1.001365e+00</td>
      <td>1.001365e+00</td>
      <td>1.001365e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.000000</td>
      <td>-1.441514e+00</td>
      <td>-1.193512e+00</td>
      <td>-1.543477e+00</td>
      <td>-1.617357e+00</td>
      <td>-2.767960e+00</td>
      <td>-1.611995e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.550000</td>
      <td>-8.504125e-01</td>
      <td>-8.499642e-01</td>
      <td>-7.666291e-01</td>
      <td>-8.750977e-01</td>
      <td>-6.396984e-01</td>
      <td>-7.997267e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23.000000</td>
      <td>-8.504125e-01</td>
      <td>-4.475220e-01</td>
      <td>-2.576598e-01</td>
      <td>-2.003166e-01</td>
      <td>-1.589748e-02</td>
      <td>1.254184e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>29.000000</td>
      <td>1.513992e+00</td>
      <td>6.714636e-01</td>
      <td>4.656124e-01</td>
      <td>7.431720e-01</td>
      <td>5.528622e-01</td>
      <td>8.248104e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>46.600000</td>
      <td>1.513992e+00</td>
      <td>2.585518e+00</td>
      <td>3.385489e+00</td>
      <td>2.632559e+00</td>
      <td>3.396661e+00</td>
      <td>1.637079e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split
```

**Info:** Our primary goal in this tutorial is to learn how to handle multicollinearity among features , hence we are not considering the **origin** variable in our features as it's a categorical feature.
{: .notice--info}


```python
X=cars_df.drop(['mpg','origin'],axis=1) 
Y=cars_df['mpg']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
```


```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(normalize=True).fit(x_train,y_train)
```


```python
print("Training score : ",linear_model.score(x_train,y_train))
```

    Training score :  0.8003438238657309
    


```python
y_pred = linear_model.predict(x_test)
```


```python
from sklearn.metrics import r2_score

print("Testing_score :",r2_score(y_test,y_pred))
```

    Testing_score : 0.8190012505093899
    

## What is Adjusted $R^2$ Score?

When we have multiple predictors/features , A better measure of how good our model is **Adjusted $R^2$ score**

The Adjusted $R^2$ score is calculated using r2_score and it is a corrected goodness of fit measure for linear models.
This is an Adjusted $R^2$ score that has been adjusted for the number of predictors/features we have used in our regression analysis.

- The Adjusted $R^2$ score increases when a new predictor/feature has been added to train our model imporves our model more than the improvement that can be expected purely due to chance.

- When we don't have highly correlated features then we can observe that Adjusted $R^2$ score is very close to our actual r2 score.



```python
def adjusted_r2(r_square,labels,features):
    adj_r_square = 1 - ((1- r_square)*(len(labels)-1))/(len(labels)- features.shape[1])
    return adj_r_square
```


```python
print("Adjusted R2 score :",adjusted_r2(r2_score(y_test,y_pred),y_test,x_test))
```

    Adjusted R2 score : 0.8056925189291979
    


```python
feature_corr=X.corr()
feature_corr
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
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cylinders</th>
      <td>1.000000</td>
      <td>0.951901</td>
      <td>0.841093</td>
      <td>0.895922</td>
      <td>-0.483725</td>
      <td>0.330754</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>0.951901</td>
      <td>1.000000</td>
      <td>0.891518</td>
      <td>0.930437</td>
      <td>-0.521733</td>
      <td>0.362976</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>0.841093</td>
      <td>0.891518</td>
      <td>1.000000</td>
      <td>0.862606</td>
      <td>-0.673175</td>
      <td>0.410110</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>0.895922</td>
      <td>0.930437</td>
      <td>0.862606</td>
      <td>1.000000</td>
      <td>-0.397605</td>
      <td>0.302727</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>-0.483725</td>
      <td>-0.521733</td>
      <td>-0.673175</td>
      <td>-0.397605</td>
      <td>1.000000</td>
      <td>-0.273762</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.330754</td>
      <td>0.362976</td>
      <td>0.410110</td>
      <td>0.302727</td>
      <td>-0.273762</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now let's explore the correlation matrix.
We discovered that there are many features which are highly correlated with **displacement**.You can see that **cylinders** , **horsepower** , **weight** are all three highly correlated with displacement.This high correlation coefficient almost at 0.9 indicates that these features are likely to be <b>colinear</b>.

Another way of saying this is **cylinders**, **horsepower**, **weight** give us the same information as **displacement**.So we dont need all of them in our regression analysis.

Using this correlation matrix let's say we want to see all those features with correlation coefficients greater than 0.8 , we can do that by below code.


```python
abs(feature_corr) > 0.8
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
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cylinders</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
trimmed_features_df = X.drop(['cylinders','horsepower','weight'],axis=1)
```


```python
trimmed_features_corr=trimmed_features_df.corr()
```


```python
trimmed_features_corr
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
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>displacement</th>
      <td>1.000000</td>
      <td>-0.521733</td>
      <td>0.362976</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>-0.521733</td>
      <td>1.000000</td>
      <td>-0.273762</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.362976</td>
      <td>-0.273762</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
abs(trimmed_features_corr) > 0.8
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
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>displacement</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Now we can check that independent features' correlation has been reduced.

## Variance Inflation Factor

Another way of selecting features which are not colinear is <b><u>Variance Inflation Factor</u></b>.This is a measure to quantify the severity of multicolinearity in an ordinary least squares regression analysis.

Variance inflation factor is a measure of the amount of multicollinearity in a set of multiple regression variables.

Variance inflation factor measures how much the behavior (variance) of an independent variable is influenced, or inflated, by its interaction/correlation with the other independent variables.


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```


```python
vif=pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
```


```python
vif['Features'] = X.columns
```


```python
vif.round(2)
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
      <th>VIF Factor</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.82</td>
      <td>cylinders</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.13</td>
      <td>displacement</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.98</td>
      <td>horsepower</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.36</td>
      <td>weight</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.50</td>
      <td>acceleration</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.24</td>
      <td>Age</td>
    </tr>
  </tbody>
</table>
</div>



- VIF = 1: Not correlated
- VIF =1-5: Moderately correlated
- VIF >5: Highly correlated


If we look at the VIF factors we can see displacement and weight are highly correlated features so let's drop it from Features.


```python
X = X.drop(['displacement','weight'], axis = 1)
```

Now again we calculate the VIF for the rest of the features


```python
vif=pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
```


```python
vif['Features'] = X.columns
vif.round(2)
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
      <th>VIF Factor</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.57</td>
      <td>cylinders</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.26</td>
      <td>horsepower</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.91</td>
      <td>acceleration</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.20</td>
      <td>Age</td>
    </tr>
  </tbody>
</table>
</div>



So now colinearity of features has been reduced using VIF.


```python
X=cars_df.drop(['mpg','origin','displacement','weight'],axis=1)
Y=cars_df['mpg']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
```


```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(normalize=True).fit(x_train,y_train)
```


```python
print("Training score : ",linear_model.score(x_train,y_train))
```

    Training score :  0.7537877265338784
    


```python
y_pred = linear_model.predict(x_test)
```


```python
from sklearn.metrics import r2_score

print("Testing_score :",r2_score(y_test,y_pred))
```

    Testing_score : 0.7159725745358863
    


```python
print("Adjusted R2 score :",adjusted_r2(r2_score(y_test,y_pred),y_test,x_test))
```

    Adjusted R2 score : 0.7037999705874243
    
