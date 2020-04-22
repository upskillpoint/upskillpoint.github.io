---
title: "T103: Filter method-Feature selection techniques in machine learning"
date: 2020-03-11
categories: [Machine Learning]
tags: [Data Pre-processing,Feature Selection,Filter Method,Anova F-Test,Chi-2 Test,Missing Value Ratio Threshold,Variance Threshold]
excerpt: "A step by step guide on how to select features using filter method"
toc: true
header:
  teaser: /assets/images/2020/03/12/filter_method.png
  image: /assets/images/2020/03/12/filter_method.png
  show_overlay_excerpt: False
mathjax: true
---

**Note:** This is a part of series on Data Preprocessing in Machine Learning you can check all tutorials here: [Embedded Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-embedded-method/), [Wrapper Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-wrapper-method/), [Filter Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-filter-method/),[Handling Multicollinearity](https://datamould.github.io/machine%20learning/2020/03/13/feature-selection-handling-multicollinearity/).
{: .notice--info}

In this tutorial we will see how we can select features using Filter feature selection method.

## Filter Methods

Filter method applies a statistical measure to assign a scoring to each feature.Then we can decide to keep or remove those features based on those scores. The methods are often univariate and consider the feature independently, or with regard to the dependent variable.

In this tutorial we will cover below approaches:

1. Missing Value Ratio Threshold
2. Variance Threshold
3. $Chi^2$ Test
4. Anova Test

### Missing Value Ratio Threshold

We will remove those features which are having missing values more than a threshold.


```python
import pandas as pd
import numpy as np
```


```python
diabetes = pd.read_csv('dataset/diabetes.csv')
diabetes.head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We know that below features can not be zero(e.g. a person's blood pressure can not be 0) hence we are imputing zeros with nan value in these features.


```python
diabetes['Glucose'].replace(0,np.nan,inplace=True)
diabetes['BloodPressure'].replace(0,np.nan,inplace=True)
diabetes['SkinThickness'].replace(0,np.nan,inplace=True)
diabetes['Insulin'].replace(0,np.nan,inplace=True)
diabetes['BMI'].replace(0,np.nan,inplace=True)
```


```python
diabetes.isnull().sum()
```




    Pregnancies                   0
    Glucose                       5
    BloodPressure                35
    SkinThickness               227
    Insulin                     374
    BMI                          11
    DiabetesPedigreeFunction      0
    Age                           0
    Outcome                       0
    dtype: int64



Now let's see for each feature what is the percentage of having missing values.


```python
diabetes['Glucose'].isnull().sum()/len(diabetes)*100
```




    0.6510416666666667




```python
diabetes['BloodPressure'].isnull().sum()/len(diabetes)*100
```




    4.557291666666666




```python
diabetes['SkinThickness'].isnull().sum()/len(diabetes)*100
```




    29.557291666666668




```python
diabetes['Insulin'].isnull().sum()/len(diabetes)*100
```




    48.69791666666667




```python
diabetes['BMI'].isnull().sum()/len(diabetes)*100
```




    1.4322916666666665



We can see that a large number of data missing in **SkinThickness**, **Insulin**.


```python
diabetes_missing_value_threshold=diabetes.dropna(thresh=int(diabetes.shape[0] * .9) ,axis=1)
diabetes_missing_value_threshold
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>218.937760</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>85.0</td>
      <td>66.0</td>
      <td>29.0</td>
      <td>70.189298</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>183.0</td>
      <td>64.0</td>
      <td>29.0</td>
      <td>269.968908</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>89.0</td>
      <td>66.0</td>
      <td>23.0</td>
      <td>94.000000</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>35.0</td>
      <td>168.000000</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>10.0</td>
      <td>101.0</td>
      <td>76.0</td>
      <td>48.0</td>
      <td>180.000000</td>
      <td>32.9</td>
      <td>0.171</td>
      <td>63.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2.0</td>
      <td>122.0</td>
      <td>70.0</td>
      <td>27.0</td>
      <td>158.815881</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>765</th>
      <td>5.0</td>
      <td>121.0</td>
      <td>72.0</td>
      <td>23.0</td>
      <td>112.000000</td>
      <td>26.2</td>
      <td>0.245</td>
      <td>30.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1.0</td>
      <td>126.0</td>
      <td>60.0</td>
      <td>29.0</td>
      <td>173.820363</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1.0</td>
      <td>93.0</td>
      <td>70.0</td>
      <td>31.0</td>
      <td>87.196731</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 9 columns</p>
</div>



Here we are keeping only those features which are having missing data less than 10%.


```python
diabetes_missing_value_threshold_features = diabetes_missing_value_threshold.drop('Outcome',axis=1)
diabetes_missing_value_threshold_label= diabetes_missing_value_threshold['Outcome']
diabetes_missing_value_threshold_features,diabetes_missing_value_threshold_label
```




    (     Pregnancies  Glucose  BloodPressure  SkinThickness     Insulin   BMI  \
     0            6.0    148.0           72.0           35.0  218.937760  33.6   
     1            1.0     85.0           66.0           29.0   70.189298  26.6   
     2            8.0    183.0           64.0           29.0  269.968908  23.3   
     3            1.0     89.0           66.0           23.0   94.000000  28.1   
     4            0.0    137.0           40.0           35.0  168.000000  43.1   
     ..           ...      ...            ...            ...         ...   ...   
     763         10.0    101.0           76.0           48.0  180.000000  32.9   
     764          2.0    122.0           70.0           27.0  158.815881  36.8   
     765          5.0    121.0           72.0           23.0  112.000000  26.2   
     766          1.0    126.0           60.0           29.0  173.820363  30.1   
     767          1.0     93.0           70.0           31.0   87.196731  30.4   
     
          DiabetesPedigreeFunction   Age  
     0                       0.627  50.0  
     1                       0.351  31.0  
     2                       0.672  32.0  
     3                       0.167  21.0  
     4                       2.288  33.0  
     ..                        ...   ...  
     763                     0.171  63.0  
     764                     0.340  27.0  
     765                     0.245  30.0  
     766                     0.349  47.0  
     767                     0.315  23.0  
     
     [768 rows x 8 columns], 0      1
     1      0
     2      1
     3      0
     4      1
           ..
     763    0
     764    0
     765    0
     766    1
     767    0
     Name: Outcome, Length: 768, dtype: int64)



### Variance Threshold


If the variance is low or close to zero, then a feature is approximately constant and will not improve the performance of the model. In that case, it should be removed.

Variance will also be very low for a feature if only a handful of observations of that feature differ from a constant value.


```python
diabetes = pd.read_csv('dataset/diabetes_cleaned.csv')
diabetes.head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>218.937760</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>85.0</td>
      <td>66.0</td>
      <td>29.0</td>
      <td>70.189298</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>183.0</td>
      <td>64.0</td>
      <td>29.0</td>
      <td>269.968908</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>89.0</td>
      <td>66.0</td>
      <td>23.0</td>
      <td>94.000000</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>35.0</td>
      <td>168.000000</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X=diabetes.drop('Outcome',axis=1)
Y=diabetes['Outcome']
```


```python
X.var(axis=0)
```




    Pregnancies                    11.354056
    Glucose                       932.425376
    BloodPressure                 153.317842
    SkinThickness                 109.767160
    Insulin                     14107.703775
    BMI                            47.955463
    DiabetesPedigreeFunction        0.109779
    Age                           138.303046
    dtype: float64



We can see that **DiabetesPedigreeFunction** variance is less so it brings little information because it is (almost) constant , this can be a justification to remove **DiabetesPedigreeFunction** column but before considering this we should scale these features because they are of different scales.


```python
from sklearn.preprocessing import minmax_scale
X_scaled_df =pd.DataFrame(minmax_scale(X,feature_range=(0,10)),columns=X.columns)
```

We have used sklearn minmax scaler here.


```python
X_scaled_df
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.529412</td>
      <td>6.709677</td>
      <td>4.897959</td>
      <td>3.043478</td>
      <td>2.740295</td>
      <td>3.149284</td>
      <td>2.344150</td>
      <td>4.833333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.588235</td>
      <td>2.645161</td>
      <td>4.285714</td>
      <td>2.391304</td>
      <td>1.018185</td>
      <td>1.717791</td>
      <td>1.165670</td>
      <td>1.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.705882</td>
      <td>8.967742</td>
      <td>4.081633</td>
      <td>2.391304</td>
      <td>3.331099</td>
      <td>1.042945</td>
      <td>2.536294</td>
      <td>1.833333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.588235</td>
      <td>2.903226</td>
      <td>4.285714</td>
      <td>1.739130</td>
      <td>1.293850</td>
      <td>2.024540</td>
      <td>0.380017</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>1.632653</td>
      <td>3.043478</td>
      <td>2.150572</td>
      <td>5.092025</td>
      <td>9.436379</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>5.882353</td>
      <td>3.677419</td>
      <td>5.306122</td>
      <td>4.456522</td>
      <td>2.289500</td>
      <td>3.006135</td>
      <td>0.397096</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>764</th>
      <td>1.176471</td>
      <td>5.032258</td>
      <td>4.693878</td>
      <td>2.173913</td>
      <td>2.044244</td>
      <td>3.803681</td>
      <td>1.118702</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>765</th>
      <td>2.941176</td>
      <td>4.967742</td>
      <td>4.897959</td>
      <td>1.739130</td>
      <td>1.502241</td>
      <td>1.635992</td>
      <td>0.713066</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>766</th>
      <td>0.588235</td>
      <td>5.290323</td>
      <td>3.673469</td>
      <td>2.391304</td>
      <td>2.217956</td>
      <td>2.433538</td>
      <td>1.157131</td>
      <td>4.333333</td>
    </tr>
    <tr>
      <th>767</th>
      <td>0.588235</td>
      <td>3.161290</td>
      <td>4.693878</td>
      <td>2.608696</td>
      <td>1.215086</td>
      <td>2.494888</td>
      <td>1.011956</td>
      <td>0.333333</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 8 columns</p>
</div>




```python
X_scaled_df.var()
```




    Pregnancies                 3.928739
    Glucose                     3.869637
    BloodPressure               1.523548
    SkinThickness               0.913109
    Insulin                     1.271218
    BMI                         2.041377
    DiabetesPedigreeFunction    2.001447
    Age                         3.841751
    dtype: float64




```python
from sklearn.feature_selection import VarianceThreshold

select_features = VarianceThreshold(threshold=1.0)
```


```python
X_variance_threshold_df=select_features.fit_transform(X_scaled_df)
X_variance_threshold_df
```




    array([[3.52941176, 6.70967742, 4.89795918, ..., 3.14928425, 2.3441503 ,
            4.83333333],
           [0.58823529, 2.64516129, 4.28571429, ..., 1.71779141, 1.16567037,
            1.66666667],
           [4.70588235, 8.96774194, 4.08163265, ..., 1.04294479, 2.53629377,
            1.83333333],
           ...,
           [2.94117647, 4.96774194, 4.89795918, ..., 1.63599182, 0.71306576,
            1.5       ],
           [0.58823529, 5.29032258, 3.67346939, ..., 2.43353783, 1.15713066,
            4.33333333],
           [0.58823529, 3.16129032, 4.69387755, ..., 2.49488753, 1.01195559,
            0.33333333]])




```python
X_variance_threshold_df=pd.DataFrame(X_variance_threshold)
```


```python
X_variance_threshold_df.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.529412</td>
      <td>6.709677</td>
      <td>4.897959</td>
      <td>2.740295</td>
      <td>3.149284</td>
      <td>2.344150</td>
      <td>4.833333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.588235</td>
      <td>2.645161</td>
      <td>4.285714</td>
      <td>1.018185</td>
      <td>1.717791</td>
      <td>1.165670</td>
      <td>1.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.705882</td>
      <td>8.967742</td>
      <td>4.081633</td>
      <td>3.331099</td>
      <td>1.042945</td>
      <td>2.536294</td>
      <td>1.833333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.588235</td>
      <td>2.903226</td>
      <td>4.285714</td>
      <td>1.293850</td>
      <td>2.024540</td>
      <td>0.380017</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>1.632653</td>
      <td>2.150572</td>
      <td>5.092025</td>
      <td>9.436379</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
def get_selected_features(raw_df,processed_df):
    selected_features=[]
    for i in range(len(processed_df.columns)):
        for j in range(len(raw_df.columns)):
            if (processed_df.iloc[:,i].equals(raw_df.iloc[:,j])):
                selected_features.append(raw_df.columns[j])
    return selected_features
```


```python
selected_features= get_selected_features(X_scaled_df,X_variance_threshold_df)
selected_features
```




    ['Pregnancies',
     'Glucose',
     'BloodPressure',
     'Insulin',
     'BMI',
     'DiabetesPedigreeFunction',
     'Age']



We can see SkinThickness feature is not selected as its variance is less.


```python
X_variance_threshold_df.columns=selected_features
selected_features_df = X_variance_threshold_df
selected_features_df
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.529412</td>
      <td>6.709677</td>
      <td>4.897959</td>
      <td>2.740295</td>
      <td>3.149284</td>
      <td>2.344150</td>
      <td>4.833333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.588235</td>
      <td>2.645161</td>
      <td>4.285714</td>
      <td>1.018185</td>
      <td>1.717791</td>
      <td>1.165670</td>
      <td>1.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.705882</td>
      <td>8.967742</td>
      <td>4.081633</td>
      <td>3.331099</td>
      <td>1.042945</td>
      <td>2.536294</td>
      <td>1.833333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.588235</td>
      <td>2.903226</td>
      <td>4.285714</td>
      <td>1.293850</td>
      <td>2.024540</td>
      <td>0.380017</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>1.632653</td>
      <td>2.150572</td>
      <td>5.092025</td>
      <td>9.436379</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>5.882353</td>
      <td>3.677419</td>
      <td>5.306122</td>
      <td>2.289500</td>
      <td>3.006135</td>
      <td>0.397096</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>764</th>
      <td>1.176471</td>
      <td>5.032258</td>
      <td>4.693878</td>
      <td>2.044244</td>
      <td>3.803681</td>
      <td>1.118702</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>765</th>
      <td>2.941176</td>
      <td>4.967742</td>
      <td>4.897959</td>
      <td>1.502241</td>
      <td>1.635992</td>
      <td>0.713066</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>766</th>
      <td>0.588235</td>
      <td>5.290323</td>
      <td>3.673469</td>
      <td>2.217956</td>
      <td>2.433538</td>
      <td>1.157131</td>
      <td>4.333333</td>
    </tr>
    <tr>
      <th>767</th>
      <td>0.588235</td>
      <td>3.161290</td>
      <td>4.693878</td>
      <td>1.215086</td>
      <td>2.494888</td>
      <td>1.011956</td>
      <td>0.333333</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 7 columns</p>
</div>



Here we are keeping only those features which are having missing data less than 10%.


```python
def generate_feature_scores_df(X,Score):
    feature_score=pd.DataFrame()
    for i in range(X.shape[1]):
        new =pd.DataFrame({"Features":X.columns[i],"Score":Score[i]},index=[i])
        feature_score=pd.concat([feature_score,new])
    return feature_score
```

### Chi-Square Test

Chi2 is a measure of dependency between two variables.
It gives us a goodness of fit measure because it measures how well an observed distribution of a particular feature fits with the distribution that is expected if two features are independent.

Scikit-Learn offers a feature selection estimator named SelectKBest which select K numbers of features based on the statistical analysis.


```python
diabetes=pd.read_csv('dataset/diabetes.csv')
```


```python
X=diabetes.drop('Outcome',axis=1)
Y=diabetes['Outcome']
```


```python
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
```


```python
X=X.astype(np.float64)
```


```python
chi2_test=SelectKBest(score_func=chi2,k=4)
chi2_model=chi2_test.fit(X,Y)
```


```python
chi2_model.scores_
```




    array([ 111.51969064, 1411.88704064,   17.60537322,   53.10803984,
           2175.56527292,  127.66934333,    5.39268155,  181.30368904])




```python
feature_score_df=generate_feature_scores_df(X,chi2_model.scores_)
feature_score_df
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
      <th>Features</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pregnancies</td>
      <td>111.519691</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Glucose</td>
      <td>1411.887041</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BloodPressure</td>
      <td>17.605373</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SkinThickness</td>
      <td>53.108040</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Insulin</td>
      <td>2175.565273</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BMI</td>
      <td>127.669343</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DiabetesPedigreeFunction</td>
      <td>5.392682</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Age</td>
      <td>181.303689</td>
    </tr>
  </tbody>
</table>
</div>



Here we can see the features and corresponding chi square scores.


```python
X_new=chi2_model.transform(X)
```


```python
X_new=pd.DataFrame(X_new)
```


```python
selected_features=get_selected_features(X,X_new)
selected_features
```




    ['Glucose', 'Insulin', 'BMI', 'Age']




```python
chi2_best_features=X[selected_features]
chi2_best_features.head()
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
      <th>Glucose</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>148.0</td>
      <td>0.0</td>
      <td>33.6</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85.0</td>
      <td>0.0</td>
      <td>26.6</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>183.0</td>
      <td>0.0</td>
      <td>23.3</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>89.0</td>
      <td>94.0</td>
      <td>28.1</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>137.0</td>
      <td>168.0</td>
      <td>43.1</td>
      <td>33.0</td>
    </tr>
  </tbody>
</table>
</div>



### Anova F-Test

The F-value scores examine the varaiance by grouping the numerical feature by the target vector, the means for each group are significantly different.


```python
from sklearn.feature_selection import f_classif,SelectPercentile
Anova_test=SelectPercentile(f_classif,percentile=80)
Anova_model= Anova_test.fit(X,Y)
```

So we will be selecting only 80% out of all the features based on the F-Score. 


```python
Anova_model.scores_
```




    array([ 39.67022739, 213.16175218,   3.2569504 ,   4.30438091,
            13.28110753,  71.7720721 ,  23.8713002 ,  46.14061124])




```python
feature_scores_df=generate_feature_scores_df(X,Anova_model.scores_)
feature_scores_df
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
      <th>Features</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pregnancies</td>
      <td>39.670227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Glucose</td>
      <td>213.161752</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BloodPressure</td>
      <td>3.256950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SkinThickness</td>
      <td>4.304381</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Insulin</td>
      <td>13.281108</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BMI</td>
      <td>71.772072</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DiabetesPedigreeFunction</td>
      <td>23.871300</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Age</td>
      <td>46.140611</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_new=Anova_model.transform(X)
```


```python
X_new=pd.DataFrame(X_new)
```


```python
selected_features=get_selected_features(X,X_new)

selected_features
```




    ['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']




```python
Anova_selected_feature_df=X[selected_features]
Anova_selected_feature_df.head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>148.0</td>
      <td>0.0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>85.0</td>
      <td>0.0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>183.0</td>
      <td>0.0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>89.0</td>
      <td>94.0</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>137.0</td>
      <td>168.0</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33.0</td>
    </tr>
  </tbody>
</table>
</div>



Now let's compare predictions of these approaches.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def buildmodel(X,Y,test_frac,model_name=''):
    x_train,x_test,y_train,y_test =train_test_split(X,Y,test_size=test_frac)
    model =LogisticRegression(solver='liblinear').fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print('Accuracy of the '+model_name+' based model is ',str(accuracy_score(y_test,y_pred)))
```


```python
buildmodel(X=diabetes_missing_value_threshold_features,Y=diabetes_missing_value_threshold_label,test_frac=.2,model_name="Missing values threshold")
```

    Accuracy of the Missing values threshold based model is  0.8116883116883117
    


```python
buildmodel(X=selected_features_df,Y=Y,test_frac=.2,model_name="Variance threshold")
```

    Accuracy of the Variance threshold based model is  0.8181818181818182
    


```python
buildmodel(X=X,Y=Y,test_frac=.2,model_name="General Logistic Regression")
```

    Accuracy of the General Logistic Regression based model is  0.7012987012987013
    


```python
buildmodel(X=chi2_best_features,Y=Y,test_frac=.2,model_name="Chi2 based")
```

    Accuracy of the Chi2 based based model is  0.7922077922077922
    


```python
buildmodel(X=Anova_selected_feature_df,Y=Y,test_frac=.2,model_name="Anova F-test based")
```

    Accuracy of the Anova F-test based based model is  0.8051948051948052
    

We can definitely see that accuracy has been improved by taking these feature selection approaches. 

You can get the notebook used in this tutorial [here](https://github.com/arupbhunia/Data-Pre-processing/blob/master/FeatureSelectionUsingMissingValueRatio_VarianceThreshold.ipynb) and dataset used [here](https://github.com/arupbhunia/Data-Pre-processing/blob/master/datasets/diabetes.csv)

Thanks for reading!