---
title: "Data Pre-processing in Machine Learning"
date: 2019-01-01
categories: [Machine Learning]
tags: [Data Pre-processing,Missing values imputer]
excerpt: "A series on Data Pre-processing in Python"
header:
  teaser: /assets/images/2019/01/01/data_preprocessing.png
  image: /assets/images/2019/01/01/data_preprocessing.png
  caption: "[Source](https://medium.com/@theCADS.com/making-decisions-with-data-the-importance-of-data-preparation-d07ee9c12768)"
  show_overlay_excerpt: False
---
Data preprocessing is an integral step in Machine Learning as the quality of data and the useful information that can be derived from it directly affects the ability of our model to learn; therefore, it is extremely important that we preprocess our data before feeding it into our model.
The concepts that I will cover in this series of article are-

1. Handling Null Values
2. Standardization
3. Handling Categorical Variables
4. Discretization
5. Dimensionality Reduction 
6. Feature Selection

Let's go through an quick example to have some insights for Handling Null Values!!


```python
import numpy as np
import pandas as pd
```


```python
diabetes_df=pd.read_csv('datasets/diabetes.csv')
diabetes_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
    Pregnancies                 768 non-null int64
    Glucose                     768 non-null int64
    BloodPressure               768 non-null int64
    SkinThickness               768 non-null int64
    Insulin                     768 non-null int64
    BMI                         768 non-null float64
    DiabetesPedigreeFunction    768 non-null float64
    Age                         768 non-null int64
    Outcome                     768 non-null int64
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    


```python
diabetes_df.head()
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




```python
class CustomImputer:
    """
    This is a helper class which returns the column names based on the column index output
    """
    def __init__(self, df_columns, input_arr):
        self.df_columns = df_columns
        self.input_arr = input_arr
        
    def get_column_names(self):
        column_names=[]
        for _ in self.input_arr:
            column_names.append(self.df_columns[_])
        return column_names
```


```python
diabetes_df.describe().transpose()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pregnancies</th>
      <td>768.0</td>
      <td>3.845052</td>
      <td>3.369578</td>
      <td>0.000</td>
      <td>1.00000</td>
      <td>3.0000</td>
      <td>6.00000</td>
      <td>17.00</td>
    </tr>
    <tr>
      <th>Glucose</th>
      <td>768.0</td>
      <td>120.894531</td>
      <td>31.972618</td>
      <td>0.000</td>
      <td>99.00000</td>
      <td>117.0000</td>
      <td>140.25000</td>
      <td>199.00</td>
    </tr>
    <tr>
      <th>BloodPressure</th>
      <td>768.0</td>
      <td>69.105469</td>
      <td>19.355807</td>
      <td>0.000</td>
      <td>62.00000</td>
      <td>72.0000</td>
      <td>80.00000</td>
      <td>122.00</td>
    </tr>
    <tr>
      <th>SkinThickness</th>
      <td>768.0</td>
      <td>20.536458</td>
      <td>15.952218</td>
      <td>0.000</td>
      <td>0.00000</td>
      <td>23.0000</td>
      <td>32.00000</td>
      <td>99.00</td>
    </tr>
    <tr>
      <th>Insulin</th>
      <td>768.0</td>
      <td>79.799479</td>
      <td>115.244002</td>
      <td>0.000</td>
      <td>0.00000</td>
      <td>30.5000</td>
      <td>127.25000</td>
      <td>846.00</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>768.0</td>
      <td>31.992578</td>
      <td>7.884160</td>
      <td>0.000</td>
      <td>27.30000</td>
      <td>32.0000</td>
      <td>36.60000</td>
      <td>67.10</td>
    </tr>
    <tr>
      <th>DiabetesPedigreeFunction</th>
      <td>768.0</td>
      <td>0.471876</td>
      <td>0.331329</td>
      <td>0.078</td>
      <td>0.24375</td>
      <td>0.3725</td>
      <td>0.62625</td>
      <td>2.42</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>768.0</td>
      <td>33.240885</td>
      <td>11.760232</td>
      <td>21.000</td>
      <td>24.00000</td>
      <td>29.0000</td>
      <td>41.00000</td>
      <td>81.00</td>
    </tr>
    <tr>
      <th>Outcome</th>
      <td>768.0</td>
      <td>0.348958</td>
      <td>0.476951</td>
      <td>0.000</td>
      <td>0.00000</td>
      <td>0.0000</td>
      <td>1.00000</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



## Findings:

1. This dataset contains various measures of patients who have diabetes disease.
2. Looking at the dataset we can say that it is an ideal candidate for regression model.
3. We can predict that whether a patient has diabetes or not using this.
4. Dataset contains 768 rows and 9 columns.
5. we can see apart from 'Outcome' column few columns has zero as their numerical measures which is practically impossible as per our knowledge because a person's blood pressure can't be zero or even BMI.This means our dataset having some missing values which are represented by zeros.So let's impute all zeros of these columns with np.nan.

Before imputing missing values we can check at which positions missing values are present in my features by using Missing Indicator.


```python
from sklearn.impute import MissingIndicator
indicator = MissingIndicator(missing_values=0)
indicator.fit_transform(diabetes_df)
indicator.features_
```




    array([0, 1, 2, 3, 4, 5, 8], dtype=int32)



Now let's instantiate an object from the helper class we have built above.


```python
df_cols=(diabetes_df.columns).tolist()
cols_with_missing_values=(indicator.features_).tolist()
imputer_obj = CustomImputer(df_cols,cols_with_missing_values)
cols=imputer_obj.get_column_names()
cols
```




    ['Pregnancies',
     'Glucose',
     'BloodPressure',
     'SkinThickness',
     'Insulin',
     'BMI',
     'Outcome']



Now let's impute zeros of these columns with NaN except Pregnancies and Outcome as these two columns can have zeros as their actual values.


```python
diabetes_df['Glucose'].replace(0,np.nan,inplace=True)
diabetes_df['BloodPressure'].replace(0,np.nan,inplace=True)
diabetes_df['SkinThickness'].replace(0,np.nan,inplace=True)
diabetes_df['Insulin'].replace(0,np.nan,inplace=True)
diabetes_df['BMI'].replace(0,np.nan,inplace=True)
```

We can mask our data to check the exact missing data point in our data


```python
from sklearn.impute import MissingIndicator
indicator=MissingIndicator(missing_values=np.nan)
mask_missing_values=indicator.fit_transform(diabetes_df)
mask_missing_values
```




    array([[False, False, False,  True, False],
           [False, False, False,  True, False],
           [False, False,  True,  True, False],
           ...,
           [False, False, False, False, False],
           [False, False,  True,  True, False],
           [False, False, False,  True, False]])



Below output tells us which columns are having missing values along with their counts.


```python
diabetes_df.isnull().sum()
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



So now we can see that only those 5 columns having null values.
Next we will be imputing these columns wtih different techniques.You can choose any one of them as per your use case or you can have a discussion with the SME.

### 1. Using Mode


```python
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
diabetes_df['Glucose'] = imp.fit_transform(diabetes_df['Glucose'].values.reshape(-1,1))
diabetes_df['Glucose']
```




    0      148.0
    1       85.0
    2      183.0
    3       89.0
    4      137.0
           ...  
    763    101.0
    764    122.0
    765    121.0
    766    126.0
    767     93.0
    Name: Glucose, Length: 768, dtype: float64



### 2. Using Mean


```python
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy="mean")
imp.fit(diabetes_df['BloodPressure'].values.reshape(-1,1))
diabetes_df['BloodPressure'] = imp.transform(diabetes_df['BloodPressure'].values.reshape(-1,1))
diabetes_df['BloodPressure']
```




    0      72.0
    1      66.0
    2      64.0
    3      66.0
    4      40.0
           ... 
    763    76.0
    764    70.0
    765    72.0
    766    60.0
    767    70.0
    Name: BloodPressure, Length: 768, dtype: float64



### 3.Using Median


```python
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy="median")
imp.fit(diabetes_df['SkinThickness'].values.reshape(-1,1))
diabetes_df['SkinThickness'] = imp.transform(diabetes_df['SkinThickness'].values.reshape(-1,1))
diabetes_df['SkinThickness']
```




    0      35.0
    1      29.0
    2      29.0
    3      23.0
    4      35.0
           ... 
    763    48.0
    764    27.0
    765    23.0
    766    29.0
    767    31.0
    Name: SkinThickness, Length: 768, dtype: float64



### 4. Using a Constant Value


```python
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy="constant",fill_value=22)
imp.fit(diabetes_df['BMI'].values.reshape(-1,1))
diabetes_df['BMI'] = imp.transform(diabetes_df['BMI'].values.reshape(-1,1))
diabetes_df['BMI']
```




    0      33.6
    1      26.6
    2      23.3
    3      28.1
    4      43.1
           ... 
    763    32.9
    764    36.8
    765    26.2
    766    30.1
    767    30.4
    Name: BMI, Length: 768, dtype: float64



Till this point we have treated the missing values with univariate imputation.Now let's use multivariate imputer for Insulin column.

### 5. Multivariate Imputation

Using this technique a value will be predicted for the missing value based on the other features.


```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp=IterativeImputer(max_iter=10000,random_state=0)
```

Once the imputer object is instantiated we will be dropping the target column so that biasing to our target variable can be ignored.


```python
diabetes_features=diabetes_df.drop('Outcome',axis=1)
diabetes_features
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
      <td>6</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89.0</td>
      <td>66.0</td>
      <td>23.0</td>
      <td>94.0</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>35.0</td>
      <td>168.0</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
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
      <td>10</td>
      <td>101.0</td>
      <td>76.0</td>
      <td>48.0</td>
      <td>180.0</td>
      <td>32.9</td>
      <td>0.171</td>
      <td>63</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122.0</td>
      <td>70.0</td>
      <td>27.0</td>
      <td>NaN</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
    </tr>
    <tr>
      <th>765</th>
      <td>5</td>
      <td>121.0</td>
      <td>72.0</td>
      <td>23.0</td>
      <td>112.0</td>
      <td>26.2</td>
      <td>0.245</td>
      <td>30</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126.0</td>
      <td>60.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93.0</td>
      <td>70.0</td>
      <td>31.0</td>
      <td>NaN</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 8 columns</p>
</div>




```python
diabetes_label=diabetes_df['Outcome']
diabetes_label
```




    0      1
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
    Name: Outcome, Length: 768, dtype: int64



Next fit and transform our features dataset to the imputer object


```python
imp.fit(diabetes_features)
```




    IterativeImputer(add_indicator=False, estimator=None,
                     imputation_order='ascending', initial_strategy='mean',
                     max_iter=10000, max_value=None, min_value=None,
                     missing_values=nan, n_nearest_features=None, random_state=0,
                     sample_posterior=False, tol=0.001, verbose=0)




```python
diabetes_features_arr=imp.transform(diabetes_features)
diabetes_features_arr
```




    array([[  6.   , 148.   ,  72.   , ...,  33.6  ,   0.627,  50.   ],
           [  1.   ,  85.   ,  66.   , ...,  26.6  ,   0.351,  31.   ],
           [  8.   , 183.   ,  64.   , ...,  23.3  ,   0.672,  32.   ],
           ...,
           [  5.   , 121.   ,  72.   , ...,  26.2  ,   0.245,  30.   ],
           [  1.   , 126.   ,  60.   , ...,  30.1  ,   0.349,  47.   ],
           [  1.   ,  93.   ,  70.   , ...,  30.4  ,   0.315,  23.   ]])




```python
diabetes_features=pd.DataFrame(diabetes_features_arr,columns = diabetes_features.columns)
diabetes_features
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
      <td>6.0</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>218.937760</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50.0</td>
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
      <td>10.0</td>
      <td>101.0</td>
      <td>76.0</td>
      <td>48.0</td>
      <td>180.000000</td>
      <td>32.9</td>
      <td>0.171</td>
      <td>63.0</td>
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
    </tr>
  </tbody>
</table>
<p>768 rows × 8 columns</p>
</div>



Now check if we have any missing values left.


```python
diabetes_features.isnull().sum()
```




    Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    dtype: int64



Voila!! we have imputed all the missing data in our dataset.


```python
diabetes_features.head()
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
      <td>6.0</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>218.937760</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50.0</td>
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
    </tr>
  </tbody>
</table>
</div>



Now let's concatenate our features dataset and label dataset to create the final cleaned 


```python
cleaned_diabetes_df=pd.concat([diabetes_features,diabetes_label],axis=1)
cleaned_diabetes_df
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




```python
diabetes.to_csv('datasets/diabetes_cleaned')
```

You can get the notebook used in this tutorial [here](https://github.com/arupbhunia/Data-Pre-processing/blob/master/CleanUsingSKImputer.ipynb) and dataset used [here](https://github.com/arupbhunia/Data-Pre-processing/blob/master/datasets/diabetes.csv)

Thanks for reading!
