---
title: "T102: Wrapper method-Feature selection techniques in machine learning"
date: 2020-03-11
categories: [Machine Learning]
tags: [Data Pre-processing,Feature Selection,Wrapper Method,RFE,Backward Elimination,Forward Elimination]
excerpt: "A step by step guide on how to select features using wrapper method"
toc: true
header:
  teaser: /assets/images/2020/03/11/wrapper_method.png
  image: /assets/images/2020/03/11/wrapper_method.png
  show_overlay_excerpt: False
mathjax: true
---

**Note:** This is a part of series on Data Preprocessing in Machine Learning you can check all tutorials here: [Embedded Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-embedded-method/), [Wrapper Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-wrapper-method/), [Filter Method](https://datamould.github.io/machine%20learning/2020/03/11/feature-selection-using-filter-method/),[Handling Multicollinearity](https://datamould.github.io/machine%20learning/2020/03/13/feature-selection-handling-multicollinearity/).
{: .notice--info}

In this tutorial we will see how we can select features using wrapper methods such as recursive feature elemination,forwward selection and backward selection where you generate models with subsets of features and find the best subset to work with based on the model's performance.

## What is wrapper method?

Wrapper methods are used to select a set of features by preparing where different combinations of features, then each combination is evaluated and compared to other combinations.Next a predictive model is used to assign a score based on model accuracy and to evaluate the combinations of these features.


```python
import pandas as pd
import numpy as np
```


```python
diabetes=pd.read_csv('dataset/diabetes.csv')
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




```python
X=diabetes.drop('Outcome',axis=1)
Y=diabetes['Outcome']
X,Y
```




    (     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
     0              6      148             72             35        0  33.6   
     1              1       85             66             29        0  26.6   
     2              8      183             64              0        0  23.3   
     3              1       89             66             23       94  28.1   
     4              0      137             40             35      168  43.1   
     ..           ...      ...            ...            ...      ...   ...   
     763           10      101             76             48      180  32.9   
     764            2      122             70             27        0  36.8   
     765            5      121             72             23      112  26.2   
     766            1      126             60              0        0  30.1   
     767            1       93             70             31        0  30.4   
     
          DiabetesPedigreeFunction  Age  
     0                       0.627   50  
     1                       0.351   31  
     2                       0.672   32  
     3                       0.167   21  
     4                       2.288   33  
     ..                        ...  ...  
     763                     0.171   63  
     764                     0.340   27  
     765                     0.245   30  
     766                     0.349   47  
     767                     0.315   23  
     
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



## Recursive Feature Elimination

Recursive Feature Elimination selects features by recursively considering smaller subsets of features by pruning the least important feature at each step.
Here models are created iteartively and in each iteration it determines the best and worst performing features and this process continues until all the features are explored.Next ranking is given on eah feature based on their elimination orde.
In the worst case, if a dataset contains N number of features RFE will do a greedy search for $N^2$ combinations of features.


```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
```


```python
model =LogisticRegression(solver='liblinear')
rfe=RFE(model,n_features_to_select=4)
```


```python
fit=rfe.fit(X,Y)
```


```python
print('Number of selected features',fit.n_features_)
print('Selected Features',fit.support_)
print('Feature rankings',fit.ranking_)
```

    Number of selected features 4
    Selected Features [ True  True False False False  True  True False]
    Feature rankings [1 1 2 4 5 1 1 3]
    


```python
def feature_ranks(X,Rank,Support):
    feature_rank=pd.DataFrame()
    for i in range(X.shape[1]):
        new =pd.DataFrame({"Features":X.columns[i],"Rank":Rank[i],'Selected':Support[i]},index=[i])
        feature_rank=pd.concat([feature_rank,new])
    return feature_rank
```


```python
feature_rank_df=feature_ranks(X,fit.ranking_,fit.support_)
feature_rank_df
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
      <th>Rank</th>
      <th>Selected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pregnancies</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Glucose</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BloodPressure</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SkinThickness</td>
      <td>4</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Insulin</td>
      <td>5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BMI</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DiabetesPedigreeFunction</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Age</td>
      <td>3</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



We can see there are four features with rank 1 ,RFE states that these are the most significant features.


```python
recursive_feature_names=feature_rank_df.loc[feature_rank_df['Selected'] == True]
```


```python
recursive_feature_names
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
      <th>Rank</th>
      <th>Selected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pregnancies</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Glucose</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BMI</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DiabetesPedigreeFunction</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
RFE_selected_features=X[recursive_feature_names['Features'].values]
RFE_selected_features.head()
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
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>33.6</td>
      <td>0.627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>26.6</td>
      <td>0.351</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>23.3</td>
      <td>0.672</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>28.1</td>
      <td>0.167</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>43.1</td>
      <td>2.288</td>
    </tr>
  </tbody>
</table>
</div>



## Forward Selection

In this feature selection technique one feature is added at a time based on the performance of the classifier till we get to the specified number of features.


```python
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
```


```python
feature_selector=SequentialFeatureSelector(RandomForestClassifier(n_estimators=10),
                                          k_features=4,
                                          forward=True,
                                          scoring='accuracy',
                                          cv=4)
features=feature_selector.fit(np.array(X),Y)
```


```python
forward_elimination_feature_names=list(X.columns[list(features.k_feature_idx_)])
forward_elimination_feature_names
```




    ['Glucose', 'BloodPressure', 'BMI', 'Age']




```python
forward_elimination_features_df=X[forward_elimination_feature_names]
forward_elimination_features_df.head()
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
      <th>BloodPressure</th>
      <th>BMI</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>148</td>
      <td>72</td>
      <td>33.6</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>66</td>
      <td>26.6</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>183</td>
      <td>64</td>
      <td>23.3</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>89</td>
      <td>66</td>
      <td>28.1</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>137</td>
      <td>40</td>
      <td>43.1</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>



## Backward Selection

In this feature selection technique one feature is removed at a time based on the performance of the classifier till we get to the specified number of features.


```python
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
```


```python
feature_selector=SequentialFeatureSelector(RandomForestClassifier(n_estimators=10),
                                          k_features=4,
                                          forward=False,
                                          scoring='accuracy',
                                          cv=4)
features=feature_selector.fit(np.array(X),Y)
```


```python
backward_elimination_feature_names=list(X.columns[list(features.k_feature_idx_)])
backward_elimination_feature_names
```




    ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']




```python
backward_elimination_features_df=X[backward_elimination_feature_names]
backward_elimination_features_df.head()
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
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>148</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>183</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>89</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>137</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>




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
buildmodel(RFE_selected_features,Y,test_frac=.2,model_name='RFE')
```

    Accuracy of the RFE based model is  0.7727272727272727
    


```python
buildmodel(forward_elimination_features_df,Y,test_frac=.2,model_name='Forward Elimination')
```

    Accuracy of the Forward Elimination based model is  0.6883116883116883
    


```python
buildmodel(backward_elimination_features_df,Y,test_frac=.2,model_name='Backward Elimination')
```

    Accuracy of the Backward Elimination based model is  0.8441558441558441

You can get the notebook used in this tutorial [here](https://github.com/arupbhunia/Data-Pre-processing/blob/master/FeatureSelectionUsingWrapperMethod.ipynb) and dataset used [here](https://github.com/arupbhunia/Data-Pre-processing/blob/master/datasets/diabetes.csv)
    
Thanks for reading!!
    
