---
layout: post
title: numpy_exercise
---

This notebook is used to do some exercises about numpy basic knowledge
you can see the details in https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

```python
# An example
```


```python
import numpy as np
```


```python
a = np.arange(15).reshape(3,5)
```


```python
a
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
# return the dimensions of the array
a.shape
```




    (3L, 5L)




```python
# return the number of dimensions(axes) of the array
a.ndim
```




    2




```python
# return the name of the type of the elements in the array
a.dtype.name
```




    'int32'




```python
# return the size in bytes of each element of the array
a.itemsize
```




    4




```python
# return the number of array, equals to the product of the elements of shape
a.size
```




    15




```python
# obviously
type(a)
```




    numpy.ndarray




```python
# create another array
b = np.array([6,7,8])
```


```python
b
```




    array([6, 7, 8])



# Summarize ways to create array


```python

```
