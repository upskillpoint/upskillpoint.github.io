---
title: "Feature scaling and transformation in machine learning"
date: 2020-03-15
categories: [Machine Learning]
tags: [Data Pre-processing,Feature Scaling,Feature Transformation,Normalization,Standardization]
excerpt: "A step by step guide on how to scale or transform features in machine learning"
toc: true
header:
  teaser: /assets/images/2020/03/15/feature_scaling.png
  image: /assets/images/2020/03/13/feature_scaling.png
  show_overlay_excerpt: False
mathjax: true  
---

In this tutorial we will learn how to scale or transform the features using different techniques. 


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
diabetes=pd.read_csv('diabetes_cleaned.csv')
```


```python
features_df= diabetes.drop('Outcome',axis = 1)
target_df=diabetes['Outcome']
```


```python
features_df.describe()
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
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>121.539062</td>
      <td>72.405184</td>
      <td>29.108073</td>
      <td>152.222767</td>
      <td>32.307682</td>
      <td>0.471876</td>
      <td>33.240885</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>30.490660</td>
      <td>12.096346</td>
      <td>8.791221</td>
      <td>97.387162</td>
      <td>6.986674</td>
      <td>0.331329</td>
      <td>11.760232</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>44.000000</td>
      <td>24.000000</td>
      <td>7.000000</td>
      <td>-17.757186</td>
      <td>18.200000</td>
      <td>0.078000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>64.000000</td>
      <td>25.000000</td>
      <td>89.647494</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.202592</td>
      <td>29.000000</td>
      <td>130.000000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>188.448695</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
    </tr>
  </tbody>
</table>
</div>



As you can see range of columns are very high for e.g. Age range is 21-81. ML models always give you better result when all of the features are in same range. So let's do that using various techniques.


## Feature Scaling and Standardization

When all features are in different range then we change the range of those features to a specific scale ,this method is called feature scaling.

- Normalization and Standardization are two specific Feature Scaling methods.

### Min Max Scaler


```python
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))
rescaled_features=scaler.fit_transform(features_df)
rescaled_features
```




    array([[0.35294118, 0.67096774, 0.48979592, ..., 0.31492843, 0.23441503,
            0.48333333],
           [0.05882353, 0.26451613, 0.42857143, ..., 0.17177914, 0.11656704,
            0.16666667],
           [0.47058824, 0.89677419, 0.40816327, ..., 0.10429448, 0.25362938,
            0.18333333],
           ...,
           [0.29411765, 0.49677419, 0.48979592, ..., 0.16359918, 0.07130658,
            0.15      ],
           [0.05882353, 0.52903226, 0.36734694, ..., 0.24335378, 0.11571307,
            0.43333333],
           [0.05882353, 0.31612903, 0.46938776, ..., 0.24948875, 0.10119556,
            0.03333333]])




```python
rescaled_diabetes = pd.DataFrame(rescaled_features, columns=features_df.columns)
rescaled_diabetes
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
      <td>0.352941</td>
      <td>0.670968</td>
      <td>0.489796</td>
      <td>0.304348</td>
      <td>0.274029</td>
      <td>0.314928</td>
      <td>0.234415</td>
      <td>0.483333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.058824</td>
      <td>0.264516</td>
      <td>0.428571</td>
      <td>0.239130</td>
      <td>0.101819</td>
      <td>0.171779</td>
      <td>0.116567</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.470588</td>
      <td>0.896774</td>
      <td>0.408163</td>
      <td>0.239130</td>
      <td>0.333110</td>
      <td>0.104294</td>
      <td>0.253629</td>
      <td>0.183333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.058824</td>
      <td>0.290323</td>
      <td>0.428571</td>
      <td>0.173913</td>
      <td>0.129385</td>
      <td>0.202454</td>
      <td>0.038002</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.163265</td>
      <td>0.304348</td>
      <td>0.215057</td>
      <td>0.509202</td>
      <td>0.943638</td>
      <td>0.200000</td>
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
      <td>0.588235</td>
      <td>0.367742</td>
      <td>0.530612</td>
      <td>0.445652</td>
      <td>0.228950</td>
      <td>0.300613</td>
      <td>0.039710</td>
      <td>0.700000</td>
    </tr>
    <tr>
      <th>764</th>
      <td>0.117647</td>
      <td>0.503226</td>
      <td>0.469388</td>
      <td>0.217391</td>
      <td>0.204424</td>
      <td>0.380368</td>
      <td>0.111870</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>765</th>
      <td>0.294118</td>
      <td>0.496774</td>
      <td>0.489796</td>
      <td>0.173913</td>
      <td>0.150224</td>
      <td>0.163599</td>
      <td>0.071307</td>
      <td>0.150000</td>
    </tr>
    <tr>
      <th>766</th>
      <td>0.058824</td>
      <td>0.529032</td>
      <td>0.367347</td>
      <td>0.239130</td>
      <td>0.221796</td>
      <td>0.243354</td>
      <td>0.115713</td>
      <td>0.433333</td>
    </tr>
    <tr>
      <th>767</th>
      <td>0.058824</td>
      <td>0.316129</td>
      <td>0.469388</td>
      <td>0.260870</td>
      <td>0.121509</td>
      <td>0.249489</td>
      <td>0.101196</td>
      <td>0.033333</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 8 columns</p>
</div>




```python
rescaled_diabetes.boxplot(figsize=(12,10),rot=45)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x164058b0>




![png]({{ site.url }}/assets/images/2020/03/15/output_10_1.png)


You can see all the data column's range is between 0-1 now.

But here is one Catch!! MinMaxScaler is very sensitive to your data so make sure your whole prediction should not be hampered , like in this case if you see it has change age's min to 0 but that is not the one we are looking for.

### Standardization


Standardization is applied on feature wise , we calculate the mean of each feature then subtract each feature value from the mean and then divide it with the standard deivation
Standardization centers mean of all our numeric features at 0 and expresses each value of the feature by the multiples of the std dev. 

This is usually preferred because it is less outlier sensitive.


```python
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
standardized_features=scaler.fit_transform(features_df)
```


```python
standardized_diabetes = pd.DataFrame(standardized_features, columns=features_df.columns)
standardized_diabetes

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
      <td>0.639947</td>
      <td>0.868403</td>
      <td>-0.033518</td>
      <td>0.670643</td>
      <td>0.685496</td>
      <td>0.185089</td>
      <td>0.468492</td>
      <td>1.425995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.844885</td>
      <td>-1.199150</td>
      <td>-0.529859</td>
      <td>-0.012301</td>
      <td>-0.842893</td>
      <td>-0.817471</td>
      <td>-0.365061</td>
      <td>-0.190672</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.233880</td>
      <td>2.017044</td>
      <td>-0.695306</td>
      <td>-0.012301</td>
      <td>1.209840</td>
      <td>-1.290106</td>
      <td>0.604397</td>
      <td>-0.105584</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.844885</td>
      <td>-1.067877</td>
      <td>-0.529859</td>
      <td>-0.695245</td>
      <td>-0.598238</td>
      <td>-0.602636</td>
      <td>-0.920763</td>
      <td>-1.041549</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.141852</td>
      <td>0.507402</td>
      <td>-2.680669</td>
      <td>0.670643</td>
      <td>0.162111</td>
      <td>1.545707</td>
      <td>5.484909</td>
      <td>-0.020496</td>
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
      <td>1.827813</td>
      <td>-0.674057</td>
      <td>0.297376</td>
      <td>2.150354</td>
      <td>0.285411</td>
      <td>0.084833</td>
      <td>-0.908682</td>
      <td>2.532136</td>
    </tr>
    <tr>
      <th>764</th>
      <td>-0.547919</td>
      <td>0.015127</td>
      <td>-0.198965</td>
      <td>-0.239949</td>
      <td>0.067744</td>
      <td>0.643403</td>
      <td>-0.398282</td>
      <td>-0.531023</td>
    </tr>
    <tr>
      <th>765</th>
      <td>0.342981</td>
      <td>-0.017691</td>
      <td>-0.033518</td>
      <td>-0.695245</td>
      <td>-0.413288</td>
      <td>-0.874760</td>
      <td>-0.685193</td>
      <td>-0.275760</td>
    </tr>
    <tr>
      <th>766</th>
      <td>-0.844885</td>
      <td>0.146400</td>
      <td>-1.026200</td>
      <td>-0.012301</td>
      <td>0.221915</td>
      <td>-0.316191</td>
      <td>-0.371101</td>
      <td>1.170732</td>
    </tr>
    <tr>
      <th>767</th>
      <td>-0.844885</td>
      <td>-0.936604</td>
      <td>-0.198965</td>
      <td>0.215347</td>
      <td>-0.668142</td>
      <td>-0.273224</td>
      <td>-0.473785</td>
      <td>-0.871374</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 8 columns</p>
</div>




```python
standardized_diabetes.boxplot(figsize=(12,10),rot=45)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1679ec30>




![png]({{ site.url }}/assets/images/2020/03/15/output_16_1.png)


You can see that all features' mean have been centered to zero and if any feature is not having many outliers then its median should not be far away from the mean.

### Normalization

Normalization converts the feature vectors to their unit norm representations , there are different types of unit norms such as
1. L1 Normalization
2. L2 Normalization
3. Max Normalization

This is not useful with data having outliers!


```python
from sklearn.preprocessing import Normalizer
```

#### L1 Normailization


```python
normalizer = Normalizer(norm='l1')
l1_normalized_features = normalizer.fit_transform(features_df)
```


```python
l1_normalized_diabetes = pd.DataFrame(l1_normalized_features, columns=features_df.columns)
l1_normalized_diabetes.head()
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
      <td>0.010635</td>
      <td>0.262335</td>
      <td>0.127622</td>
      <td>0.062039</td>
      <td>0.388074</td>
      <td>0.059557</td>
      <td>0.001111</td>
      <td>0.088627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.003235</td>
      <td>0.274956</td>
      <td>0.213495</td>
      <td>0.093809</td>
      <td>0.227047</td>
      <td>0.086045</td>
      <td>0.001135</td>
      <td>0.100278</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.013116</td>
      <td>0.300029</td>
      <td>0.104928</td>
      <td>0.047546</td>
      <td>0.442615</td>
      <td>0.038200</td>
      <td>0.001102</td>
      <td>0.052464</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.003103</td>
      <td>0.276169</td>
      <td>0.204799</td>
      <td>0.071369</td>
      <td>0.291684</td>
      <td>0.087195</td>
      <td>0.000518</td>
      <td>0.065163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.298873</td>
      <td>0.087262</td>
      <td>0.076355</td>
      <td>0.366502</td>
      <td>0.094025</td>
      <td>0.004991</td>
      <td>0.071991</td>
    </tr>
  </tbody>
</table>
</div>




```python
l1_normalized_diabetes.iloc[0]
```




    Pregnancies                 0.010635
    Glucose                     0.262335
    BloodPressure               0.127622
    SkinThickness               0.062039
    Insulin                     0.388074
    BMI                         0.059557
    DiabetesPedigreeFunction    0.001111
    Age                         0.088627
    Name: 0, dtype: float64



Every row in your dataset is a feature vector and normalization is a technique to convert those feature vector by their unit magnitude
there are different types of unit magnitudes here we have converted using L1 unit magnitude.

In L1 normalization summation of absolute values of these normalized features is 1.


```python
l1_normalized_diabetes.iloc[0].abs().sum()
```




    1.0



#### L2 Normalization

In L2 normalization every feature vector or records in your dataset will be converted to their L2 unit magnitude and sum of the individual features' square will be 1


```python
normalizer = Normalizer(norm='l2')
l2_normalized_features = normalizer.fit_transform(features_df)
l2_normalized_diabetes = pd.DataFrame(l2_normalized_features, columns=features_df.columns)
l2_normalized_diabetes.head()
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
      <td>0.021225</td>
      <td>0.523547</td>
      <td>0.254698</td>
      <td>0.123812</td>
      <td>0.774487</td>
      <td>0.118859</td>
      <td>0.002218</td>
      <td>0.176874</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.007251</td>
      <td>0.616359</td>
      <td>0.478585</td>
      <td>0.210287</td>
      <td>0.508963</td>
      <td>0.192884</td>
      <td>0.002545</td>
      <td>0.224790</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.023805</td>
      <td>0.544535</td>
      <td>0.190439</td>
      <td>0.086292</td>
      <td>0.803320</td>
      <td>0.069332</td>
      <td>0.002000</td>
      <td>0.095219</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.006612</td>
      <td>0.588467</td>
      <td>0.436392</td>
      <td>0.152076</td>
      <td>0.621527</td>
      <td>0.185797</td>
      <td>0.001104</td>
      <td>0.138852</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.596386</td>
      <td>0.174127</td>
      <td>0.152361</td>
      <td>0.731335</td>
      <td>0.187622</td>
      <td>0.009960</td>
      <td>0.143655</td>
    </tr>
  </tbody>
</table>
</div>




```python
l2_normalized_diabetes.iloc[0].pow(2).sum()
```




    0.9999999999999997



#### Maximum Normalization

Now let's talk about Maximum normalization here the maximum value of a feature vector is converted to 1 and other values of that feature vector will be converted in terms of this maximum.


```python
normalizer = Normalizer(norm='max')
max_normalized_features = normalizer.fit_transform(features_df)
print(type(max_normalized_features))
max_normalized_diabetes = pd.DataFrame(max_normalized_features, columns=features_df.columns)
max_normalized_diabetes.head()
```

    <class 'numpy.ndarray'>
    




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
      <td>0.027405</td>
      <td>0.675991</td>
      <td>0.328861</td>
      <td>0.159863</td>
      <td>1.000000</td>
      <td>0.153468</td>
      <td>0.002864</td>
      <td>0.228375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.011765</td>
      <td>1.000000</td>
      <td>0.776471</td>
      <td>0.341176</td>
      <td>0.825756</td>
      <td>0.312941</td>
      <td>0.004129</td>
      <td>0.364706</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.029633</td>
      <td>0.677856</td>
      <td>0.237064</td>
      <td>0.107420</td>
      <td>1.000000</td>
      <td>0.086306</td>
      <td>0.002489</td>
      <td>0.118532</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.010638</td>
      <td>0.946809</td>
      <td>0.702128</td>
      <td>0.244681</td>
      <td>1.000000</td>
      <td>0.298936</td>
      <td>0.001777</td>
      <td>0.223404</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.815476</td>
      <td>0.238095</td>
      <td>0.208333</td>
      <td>1.000000</td>
      <td>0.256548</td>
      <td>0.013619</td>
      <td>0.196429</td>
    </tr>
  </tbody>
</table>
</div>



if you look at the above df you can see one feature in every record is transformed into 1 and other features are represented in terms of this max.

# Binarizer:

Now sometimes it may be required that we would want to discretize our numerical features there we can use binarizer.
In binarizer we provide a threshold value for each feature and it will convert all values which is less than the threshold to zero and all values which is greater than the threshold to 1. 


```python
scaler=Binarizer(threshold=float((features_df[['Pregnancies']]).mean()))
binarized_features=scaler.fit_transform(features_df[['Pregnancies']])
```


```python
from sklearn.preprocessing import Binarizer
for i in range(1,features_df.shape[1]):
    scaler=Binarizer(threshold=float(features_df[features_df.columns[i]].mean())). \
                                    fit(features_df[[features_df.columns[i]]])
    new_binarized_feature = scaler.transform(features_df[[features_df.columns[i]]])
    binarized_features = np.concatenate((binarized_features,new_binarized_feature),axis=1)
```


```python
binarized_diabetes = pd.DataFrame(binarized_features, columns=features_df.columns)
binarized_diabetes.head(20)
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
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



You can see all vectors have been represented using zero or 1
Now that we have transformed our data using different techniques let's do some classification now.

Now lets build a classification model and see the differences:


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


```python
def buildmodel(X,Y,test_frac):
    x_train,x_test,y_train,y_test =train_test_split(X,Y,test_size=test_frac)
    model =LogisticRegression(solver='liblinear').fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print('Accuracy of the model is',accuracy_score(y_test,y_pred))
```


```python
buildmodel(rescaled_diabetes,target_df,test_frac=.2)#using MinMaxScaler
```

    Accuracy of the model is 0.7857142857142857
    


```python
buildmodel(standardized_diabetes,target_df,test_frac=.2)#using StandardScaler
```

    Accuracy of the model is 0.7662337662337663
    


```python
buildmodel(l1_normalized_features,target_df,test_frac=.2)
```

    Accuracy of the model is 0.6103896103896104
    


```python
buildmodel(l2_normalized_features,target_df,test_frac=.2)
```

    Accuracy of the model is 0.6623376623376623
    


```python
buildmodel(max_normalized_features,target_df,test_frac=.2)
```

    Accuracy of the model is 0.7337662337662337
    


```python
buildmodel(binarized_features,target_df,test_frac=.2)
```

    Accuracy of the model is 0.6753246753246753
    
