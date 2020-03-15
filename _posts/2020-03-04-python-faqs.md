---
title: "Things every Python developer should know"
date: 2020-03-04
categories: [Python FAQ]
tags: [Python]
excerpt: "This blog answers some frequently asked questions on Python"
---


### What is the difference between arguments and parameters?


Parameters are defined by the names that appear in a function definition, whereas arguments are the values actually passed to a function when calling it. Parameters define what types of arguments a function can accept. For example, given the function definition:
```python
def func(x, y=None, **kwargs):
    pass
```

x, y and kwargs are parameters of func. However, when calling func, for example:

```python
func(2, y=11, country='India')
```
the values 2, 11, and India are arguments.

### Is there a tool to help find bugs or perform static analysis?

Yes.
Please check out this blog [Pylint for linting your code.](https://datamould.github.io/python%20static%20testing/2020/03/01/step-by-step-guide-pylint-pycharm-integration/)

###  What is the purpose of “pip install --user {packagename}”? 

pip defaults to installing Python packages to a system directory (such as /usr/local/lib/python3.4). This requires root access.

--user makes pip install packages in your home directory instead, which doesn't require any special privileges.

### Where can I locate the Python site-packages directory?

There are two types of site-packages directories, global and per user.

1. Global site-packages ("dist-packages") directories are listed in sys.path when you run:

<pre>
python -m site
</pre>

2. The per user site-packages directory (PEP 370) is where Python installs your local packages:

<pre>
python -m site --user-site
</pre>

### How to get the details of installed python packages?

Simply use below command: 
<pre>
pip show {package_name}
</pre>


This blog will be continued with more interesting questions!!