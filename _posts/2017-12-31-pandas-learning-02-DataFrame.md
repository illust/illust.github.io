---
layout: post
title: Pandas学习笔记02
categories: [blog]
excerpt: Sunny Day!
---

# Pandas学习笔记02

### 1. pandas.DataFrame结构
pandas.DataFrame(data,index,columns,dtype,copy)

### 2. 创建DataFrame
Pandas的DataFrame可以使用以下输入创建：
- 列表
- 字典
- Series
- Numpy模块的ndarray
- 其他DataFrame

### 3. 创建空DataFrame


```python
import pandas as pd
df = pd.DataFrame()
print df
```

    Empty DataFrame
    Columns: []
    Index: []


### 4. 从列表创建一个DataFrame


```python
import pandas as pd
data = [1,2,3,4,5]
df = pd.DataFrame(data)
print df
```

       0
    0  1
    1  2
    2  3
    3  4
    4  5



```python
# 列名分别指定为'Name','Age'
import pandas as pd
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'])
print df
```

         Name  Age
    0    Alex   10
    1     Bob   12
    2  Clarke   13


### 5. 从ndarrays/列表字典创建一个DataFrame


```python
# 所有数组长度必须相等
import pandas as pd
data = {'Name':['Tom','Jack','Steve','Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
print df
```

       Age   Name
    0   28    Tom
    1   34   Jack
    2   29  Steve
    3   42  Ricky


### 6. 从字典列表创建一个DataFrame
字典关键字默认作为列名


```python
# 可以看出，c列在索引为first处没有指定值，系统自动设为NaN(Not a Number)
import pandas as pd
data = [{'a':1,'b':2},{'a':5,'b':10,'c':20}]
df = pd.DataFrame(data,index=['first','second'])
print df
```

            a   b     c
    first   1   2   NaN
    second  5  10  20.0



```python
# 另一个例子，列表中的两个元素均为字典类型
import pandas as pd
data = [{'a':1,'b':2},{'a':5,'b':10,'c':20}]

# 两个列名称与字典关键字相同
df1 = pd.DataFrame(data,index=['first','second'],columns=['a','b'])
print df1

# 只有一个列名与字典关键字相同，另一个为未知名称
df2 = pd.DataFrame(data,index=['first','second'],columns=['a','b1'])
print df2
```

            a   b
    first   1   2
    second  5  10
            a  b1
    first   1 NaN
    second  5 NaN


### 7. 从Series字典创建一个DataFrame


```python
import pandas as pd
d = {'one':pd.Series([1,2,3],index=['a','b','c']),
    'two':pd.Series([1,2,3,4],index=['a','b','c','d'])}
df = pd.DataFrame(d)
print df
```

       one  two
    a  1.0    1
    b  2.0    2
    c  3.0    3
    d  NaN    4


### 8. 列选择
现在开始通过一些实例了解列选择，添加以及删除


```python
# 直接通过列名进行索引
import pandas as pd
d = {'one':pd.Series([1,2,3],index=['a','b','c']),
    'two':pd.Series([1,2,3,4],index=['a','b','c','d'])}
df = pd.DataFrame(d)
print df['one']
```

    a    1.0
    b    2.0
    c    3.0
    d    NaN
    Name: one, dtype: float64


### 9. 列添加


```python
import pandas as pd
d = {'one':pd.Series([1,2,3],index=['a','b','c']),
    'two':pd.Series([1,2,3,4],index=['a','b','c','d'])}
df = pd.DataFrame(d)

# 通过传入一个新的Series到已有的DataFrame中，用来添加新的一列
# 注意这种形式，可以直接对DataFrame新的列名索引进行Series赋值操作
print "Adding a new column by passing as Series:"
df['three'] = pd.Series([10,20,30],index=['a','b','c'])
print df

print "Adding a new column using the existing columns in DataFrame:"
df['four'] = df['one'] + df['three']
print df
```

    Adding a new column by passing as Series:
       one  two  three
    a  1.0    1   10.0
    b  2.0    2   20.0
    c  3.0    3   30.0
    d  NaN    4    NaN
    Adding a new column using the existing columns in DataFrame:
       one  two  three  four
    a  1.0    1   10.0  11.0
    b  2.0    2   20.0  22.0
    c  3.0    3   30.0  33.0
    d  NaN    4    NaN   NaN


### 10. 列删除


```python
import pandas as pd
d = {'one':pd.Series([1,2,3],index=['a','b','c']),
    'two':pd.Series([1,2,3,4],index=['a','b','c','d']),
    'three':pd.Series([10,20,30],index=['a','b','c'])}
df = pd.DataFrame(d)
print "Our dataframe is:"
print df

# 使用del函数进行列删除操作
print "Deleting the first column using DEL function:"
del df['one']
print df

# 使用pop函数进行列删除操作
print "Deleting another column using POP function:"
df.pop('two')
print df
```

    Our dataframe is:
       one  three  two
    a  1.0   10.0    1
    b  2.0   20.0    2
    c  3.0   30.0    3
    d  NaN    NaN    4
    Deleting the first column using DEL function:
       three  two
    a   10.0    1
    b   20.0    2
    c   30.0    3
    d    NaN    4
    Deleting another column using POP function:
       three
    a   10.0
    b   20.0
    c   30.0
    d    NaN


### 11. 行选择，添加和删除
能够通过传递行标签到loc函数选择行


```python
import pandas as pd
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
print df.loc['b']
```

    one    2.0
    two    2.0
    Name: b, dtype: float64


可以通过将整数位置传递给iloc函数来选择行


```python
import pandas as pd
d = {'one':pd.Series([1,2,3],index=['a','b','c']),
    'two':pd.Series([1,2,3,4],index=['a','b','c','d'])}
df = pd.DataFrame(d)

print df.iloc[2]
```

    one    3.0
    two    3.0
    Name: c, dtype: float64


行切片


```python
import pandas as pd
d = {'one':pd.Series([1,2,3],index=['a','b','c']),
    'two':pd.Series([1,2,3,4],index=['a','b','c','d'])}

df = pd.DataFrame(d)
print df[2:4]
```

       one  two
    c  3.0    3
    d  NaN    4


添加行


```python
import pandas as pd
df = pd.DataFrame([[1,2],[3,4]],columns=['a','b'])
df2 = pd.DataFrame([[5,6],[7,8]],columns=['a','b'])

df = df.append(df2)
print df
```

       a  b
    0  1  2
    1  3  4
    0  5  6
    1  7  8


删除行


```python
import pandas as pd
df = pd.DataFrame([[1,2],[3,4]],columns=['a','b'])
df2 = pd.DataFrame([[5,6],[7,8]],columns=['a','b'])

print df,'\n'
df = df.append(df2)
print df,'\n'
# 对标签0进行删除，注意标签为0的可能存在多行
df = df.drop(0)
print df
```

       a  b
    0  1  2
    1  3  4

       a  b
    0  1  2
    1  3  4
    0  5  6
    1  7  8

       a  b
    1  3  4
    1  7  8



```python

```
