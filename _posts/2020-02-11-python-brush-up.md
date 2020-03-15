---
title: "Brush up your Python knowledge"
date: 2020-02-11
categories: [Jupyter Tutorial, Python]
tags: [machine-learning, data science, messy data]
excerpt: "A tutorial to brush up python...."
header:
  teaser: /assets/images/2020/02/11/python_banner.jpg
  overlay_image: /assets/images/2020/02/11/python_banner.jpg
  caption: "Photo credit: [**NIBMWorldWide**](https://www.nibmworldwide.com/)"
  actions:
    - label: "Python Tutorial"
      url: "https://docs.python.org/3/tutorial/"
  show_overlay_excerpt: False
---

This notebook will be a short review of key concepts in python. The goal of this notebook is to jog your memory and refresh concepts.  

#### Table of contents
* Jupyter notebook
* Libraries
* Plotting
* Pandas DataFrame manipulation
* Unit testing
* Randomness and reproducibility

## Jupyter notebook
Straight from the [Juptyer website](http://jupyter.org/): "The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more."  

To run code in a cell, you can press the run tab, or press control + enter. The 'Kernel' tab is quite useful. If you find your code is stuck running (maybe you wrote an infinite loop), you can go to 'Kernel' -> 'Interrupt' to force quit. 

A few useful keyboard shortcuts:
* Run cell, select below: shift + enter
* Run cell: ctrl + enter
* Run cell, insert below: option + enter

By pressing the 'esc' key, you enter command mode (the colored border around the currently selected cell should change from green to blue). Once in command mode, you can use these shortcuts:
* Insert cell above: a
* Insert cell below: b
* Copy cell: c
* Paste cell: v
* Delete selected cell(s): d d
* Change selected cell to markdown: m
* Change selected cell to code: y

To exit command mode, click anywhere in a cell or press enter.

And of course, don't forget the ever useful:
* Save file: command + s

## Libraries
There are a few libraries that you will use almost always:
* Numpy
* Pandas
* Matplotlib or
* Seaborn

The key points to remember are how to import these libraries and their standard import names, as well as their main uses. 

Numpy (from the [Numpy website](http://www.numpy.org/)):  
NumPy is the fundamental package for scientific computing with Python. It contains among other things:

* a powerful N-dimensional array object
* sophisticated (broadcasting) functions
* tools for integrating C/C++ and Fortran code
* useful linear algebra, Fourier transform, and random number capabilities

Pandas (from the [Pandas website](https://pandas.pydata.org/)):  
high-performance, easy-to-use data structures and data analysis tools for the Python programming language

Both the Matplotlib and Seaborn libraries are for creating graphs. 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Plotting


```python
# Load in the data set
tips_data = sns.load_dataset("tips")
```

### Plot a histogram of the tips


```python
# with seaborn
sns.distplot(tips_data["tip"], kde = False, bins=10).set_title("Histogram of Total Tip")
plt.show()
```


![png]({{ site.url }}/assets/images/2020/02/11/output_7_0.png)



```python
# with matplotlib
plt.hist(tips_data['tip'], bins=10)
plt.title("Histogram of Total Tip")
plt.show()
```


![png]({{ site.url }}/assets/images/2020/02/11/output_8_0.png)


### Create a boxplot of the total bill amounts


```python
# with seaborn
sns.boxplot(tips_data["total_bill"]).set_title("Box plot of the Total Bill")
plt.show()
```


![png]({{ site.url }}/assets/images/2020/02/11/output_10_0.png)



```python
# with matplotlib
plt.boxplot(tips_data["total_bill"])
plt.title("Box plot of the Total Bill")
plt.show()
```


![png]({{ site.url }}/assets/images/2020/02/11/output_11_0.png)


## Pandas DataFrame manipulation


```python
# Import NHANES 2015-2016 data
df = pd.read_csv("nhanes_2015_2016.csv")
```


```python
# look at top 3 rows
df.head(3)
```

#### Pick columns by name


```python
df['SEQN'].head()
```




    0    83732
    1    83733
    2    83734
    3    83735
    4    83736
    Name: SEQN, dtype: int64



#### Pick columns and rows by index name


```python
df.loc[[0, 1], ['SEQN', 'RIAGENDR']]
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
      <th>SEQN</th>
      <th>RIAGENDR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>83732</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>83733</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Pick columns and rows by index location


```python
df.iloc[[1,2], [0,5]]
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
      <th>SEQN</th>
      <th>RIAGENDR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>83733</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>83734</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Unit testing
This is the idea that you should run complicated code on a simple test case that you know that outcome of. If your code outputs something you did not expect, then you know there is an error somewhere that must be fixed. When working with large datasets, it is easy to get reasonable output that is actually measuring something different than you wanted. 

### Example
Perhaps you want to take the mean of the first row.


```python
df = pd.DataFrame({'col1':[1, 2, 3], 'col2':[3, 4, 5]})
df
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.mean()[0]
```




    2.0



This looks correct, but lets on a DataFrame that doesn't have the same mean for the first row and the first column.


```python
df = pd.DataFrame({'col1':[1, 2, 3], 'col2':[6, 7, 8]})
df
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.mean()[0]
```




    2.0



Looks like this is actually taking the mean of the first row. Doing a simple test, we found an error that would have been much harder to spot had our DataFrame been 100,000 rows and 300 columns. 


```python
# Use the argument 'axis=1' to specify that we are taking the mean of the columns. 
# Use 'axis=0' (which as we can see is the default) to take to mean of the rows
df.mean(axis=1)[0]
```




    3.5



## Randomness and reproducibility

In Python, we refer to randomness as the ability to generate data, strings, or, more generally, numbers at random.

However, when conducting analysis it is important to consider reproducibility. If we are creating random data, how can we enable reproducible analysis?

We do this by utilizing pseudo-random number generators (PRNGs). PRNGs start with a random number, known as the seed, and then use an algorithm to generate a psuedo-random sequence based on it.

This means that we can replicate the output of a random number generator in python simply by knowing which seed was used.

We can showcase this by using the functions in the python library random.


```python
import random
```


```python
random.seed(1234)
random.random()
```




    0.9664535356921388




The random library includes standard distributions that may come in handy


```python
# Uniform distribution
random.uniform(25,50)
```




    36.01831497938382




```python
mu = 0

sigma = 1

random.normalvariate(mu, sigma)
```




    1.8038006216944658



## List comprehension
List comprehensions allow you to easy create lists. They follow the format:
```
my_list = [expression(i) for i in input list]
```
For example, if you wanted to plot the sin curve from -$\pi$ to $\pi$:


```python
x = np.linspace(-np.pi, np.pi, 100) # create a list of 100 equally spaced points between -pi and pi
x
```




    array([-3.14159265, -3.07812614, -3.01465962, -2.9511931 , -2.88772658,
           -2.82426006, -2.76079354, -2.69732703, -2.63386051, -2.57039399,
           -2.50692747, -2.44346095, -2.37999443, -2.31652792, -2.2530614 ,
           -2.18959488, -2.12612836, -2.06266184, -1.99919533, -1.93572881,
           -1.87226229, -1.80879577, -1.74532925, -1.68186273, -1.61839622,
           -1.5549297 , -1.49146318, -1.42799666, -1.36453014, -1.30106362,
           -1.23759711, -1.17413059, -1.11066407, -1.04719755, -0.98373103,
           -0.92026451, -0.856798  , -0.79333148, -0.72986496, -0.66639844,
           -0.60293192, -0.53946541, -0.47599889, -0.41253237, -0.34906585,
           -0.28559933, -0.22213281, -0.1586663 , -0.09519978, -0.03173326,
            0.03173326,  0.09519978,  0.1586663 ,  0.22213281,  0.28559933,
            0.34906585,  0.41253237,  0.47599889,  0.53946541,  0.60293192,
            0.66639844,  0.72986496,  0.79333148,  0.856798  ,  0.92026451,
            0.98373103,  1.04719755,  1.11066407,  1.17413059,  1.23759711,
            1.30106362,  1.36453014,  1.42799666,  1.49146318,  1.5549297 ,
            1.61839622,  1.68186273,  1.74532925,  1.80879577,  1.87226229,
            1.93572881,  1.99919533,  2.06266184,  2.12612836,  2.18959488,
            2.2530614 ,  2.31652792,  2.37999443,  2.44346095,  2.50692747,
            2.57039399,  2.63386051,  2.69732703,  2.76079354,  2.82426006,
            2.88772658,  2.9511931 ,  3.01465962,  3.07812614,  3.14159265])




```python
# Here's our list comprehension. For each point in x, we want y=sin(x)
y = [np.sin(value) for value in x] 
# let looks at just the first 5 elements of y
y[:5]
```




    [-1.2246467991473532e-16,
     -0.06342391965656484,
     -0.12659245357374938,
     -0.1892512443604105,
     -0.2511479871810793]




```python
# It doesn't really matter what word you use to represent a value in the input list, 
# the following will built the same y
y = [np.sin(i) for i in x]
y[:5]
```




    [-1.2246467991473532e-16,
     -0.06342391965656484,
     -0.12659245357374938,
     -0.1892512443604105,
     -0.2511479871810793]




```python
plt.plot(x,y)
plt.show()
```


![png]({{ site.url }}/assets/images/2020/02/11/output_41_0.png)

