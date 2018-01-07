
## pandas学习笔记04

## pandas基本函数
### 1. Series基本函数功能
| 编号 | 方法 | 描述 |
| :---- | :----: | :---- |
| 1 | axes | 返回行坐标标签列表 |
| 2 | dtype | 返回对象dtype |
| 3 | empty | 若Series为空，返回true |
| 4 | ndim | 根据定义1返回基本数据的维度数 |
| 5 | size | 返回基本数据元素个数 |
| 6 | values | 将Series返回为ndarray |
| 7 | head() | 返回前n行 |
| 8 | tail() | 返回倒数n行 |

##### 创建一个Series，观察以上表格属性操作


```python
import pandas as pd
import numpy as np

s = pd.Series(np.random.randn(4))
print s
```

    0   -1.012016
    1    0.288053
    2   -0.342121
    3    1.177446
    dtype: float64
    

##### axes


```python
import pandas as pd
import numpy as np

s = pd.Series(np.random.randn(4))
print "The axes are:"
print s.axes
```

    The axes are:
    [RangeIndex(start=0, stop=4, step=1)]
    

##### empty


```python
import pandas as pd
import numpy as np

s = pd.Series(np.random.randn(4))
print "Is the Object empty?"
print s.empty
```

    Is the Object empty?
    False
    

##### ndim


```python
import pandas as pd
import numpy as np

s = pd.Series(np.random.randn(4))
print s

print "The dimensions of the object:"
print s.ndim
```

    0   -1.879772
    1   -0.682344
    2    1.697040
    3   -0.983682
    dtype: float64
    The dimensions of the object:
    1
    

##### size


```python
import pandas as pd
import numpy as np

s = pd.Series(np.random.randn(2))
print s
print "The size of the object:"
print s.size
```

    0   -0.387747
    1   -0.240524
    dtype: float64
    The size of the object:
    2
    

##### values


```python
import pandas as pd
import numpy as np

# create a series with 4 random numbers
s = pd.Series(np.random.randn(4))
print s 

print "The actual data series is:"
print s.values
```

    0   -0.800398
    1   -1.046600
    2    0.224018
    3   -0.919266
    dtype: float64
    The actual data series is:
    [-0.80039821 -1.04660011  0.22401843 -0.91926567]
    

##### Head & Tail


```python
import pandas as pd
import numpy as np

# create a series with 4 random numbers
s = pd.Series(np.random.randn(4))
print "The original series is:"
print s

print "The first two rows of the data series:"
print s.head(2)

print "The last two rows of the data series:"
print s.tail(2)
```

    The original series is:
    0    0.511740
    1   -1.460081
    2    1.757649
    3    0.791180
    dtype: float64
    The first two rows of the data series:
    0    0.511740
    1   -1.460081
    dtype: float64
    The last two rows of the data series:
    2    1.757649
    3    0.791180
    dtype: float64
    

### 2. DataFrame基本函数功能
| 编号 | 方法 | 描述 |
| :---- | :----: | :---- |
| 1 | T | 转置行和列 |
| 2 | axes | 返回带有行轴标签和列轴标签的列表作为唯一成员 |
| 3 | dtypes | 返回此对象中的dtypes |
| 4 | empty | 如果NDFrame完全是空的，则为真[无项目];如果任何轴的长度为0 |
| 5 | ndim | 轴/数组维度的数量 |
| 6 | shape | 返回表示DataFrame维度的元组 |
| 7 | size | NDFrame中的元素数目 |
| 8 | values | NDFrame的Numpy表示 |
| 9 | head() | 返回前n行 |
| 10 | tail() | 返回倒数n行 |

##### 创建一个DataFrame，观察以上表格属性操作


```python
import pandas as pd
import numpy as np

# create a dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
    'Age':pd.Series([25,26,25,23,30,29,23]),
    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

# create a DataFrame
df = pd.DataFrame(d)
print "Our data series is:"
print df
```

    Our data series is:
       Age   Name  Rating
    0   25    Tom    4.23
    1   26  James    3.24
    2   25  Ricky    3.98
    3   23    Vin    2.56
    4   30  Steve    3.20
    5   29  Smith    4.60
    6   23   Jack    3.80
    

##### 转置


```python
import pandas as pd
import numpy as np

# Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

# create a DataFrame
df = pd.DataFrame(d)
print "The transpose of the data series is:"
print df.T
```

    The transpose of the data series is:
               0      1      2     3      4      5     6
    Age       25     26     25    23     30     29    23
    Name     Tom  James  Ricky   Vin  Steve  Smith  Jack
    Rating  4.23   3.24   3.98  2.56    3.2    4.6   3.8
    

##### axes


```python
import pandas as pd
import numpy as np

# Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

# create a DataFrame
df = pd.DataFrame(d)
print "Row axis labels and column axis labels are:"
print df
print df.axes
```

    Row axis labels and column axis labels are:
       Age   Name  Rating
    0   25    Tom    4.23
    1   26  James    3.24
    2   25  Ricky    3.98
    3   23    Vin    2.56
    4   30  Steve    3.20
    5   29  Smith    4.60
    6   23   Jack    3.80
    [RangeIndex(start=0, stop=7, step=1), Index([u'Age', u'Name', u'Rating'], dtype='object')]
    

##### dtype


```python
import pandas as pd
import numpy as np

# Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

# create a DataFrame
df = pd.DataFrame(d)
print "The data types of each column are:"
print df.dtypes
```

    The data types of each column are:
    Age         int64
    Name       object
    Rating    float64
    dtype: object
    

##### empty


```python
import pandas as pd
import numpy as np

# Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

# create a DataFrame
df = pd.DataFrame(d)
print "Is the object empty?"
print df.empty
```

    Is the object empty?
    False
    

##### ndim


```python
import pandas as pd
import numpy as np

# Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

# create a DataFrame
df = pd.DataFrame(d)
print "Our object is:"
print df
print "The dimension of the object is:"
print df.ndim
```

    Our object is:
       Age   Name  Rating
    0   25    Tom    4.23
    1   26  James    3.24
    2   25  Ricky    3.98
    3   23    Vin    2.56
    4   30  Steve    3.20
    5   29  Smith    4.60
    6   23   Jack    3.80
    The dimension of the object is:
    2
    

##### shape


```python
import pandas as pd
import numpy as np

# Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

# create a DataFrame
df = pd.DataFrame(d)
print "Our object is:"
print df
print "The shape of the object is:"
print df.shape
```

    Our object is:
       Age   Name  Rating
    0   25    Tom    4.23
    1   26  James    3.24
    2   25  Ricky    3.98
    3   23    Vin    2.56
    4   30  Steve    3.20
    5   29  Smith    4.60
    6   23   Jack    3.80
    The shape of the object is:
    (7, 3)
    

##### size


```python
import pandas as pd
import numpy as np

# Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

# create a DataFrame
df = pd.DataFrame(d)
print "Our object is:"
print df
print "The total number of elements in our object is:"
print df.size
```

    Our object is:
       Age   Name  Rating
    0   25    Tom    4.23
    1   26  James    3.24
    2   25  Ricky    3.98
    3   23    Vin    2.56
    4   30  Steve    3.20
    5   29  Smith    4.60
    6   23   Jack    3.80
    The total number of elements in our object is:
    21
    

##### values
返回DataFrame中的实际数据作为NDarray


```python
import pandas as pd
import numpy as np
 
#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

# create a DataFrame
df = pd.DataFrame(d)
print "Our object is:"
print df
print "The actual data in our data frame is:"
print df.values
```

    Our object is:
       Age   Name  Rating
    0   25    Tom    4.23
    1   26  James    3.24
    2   25  Ricky    3.98
    3   23    Vin    2.56
    4   30  Steve    3.20
    5   29  Smith    4.60
    6   23   Jack    3.80
    The actual data in our data frame is:
    [[25L 'Tom' 4.23]
     [26L 'James' 3.24]
     [25L 'Ricky' 3.98]
     [23L 'Vin' 2.56]
     [30L 'Steve' 3.2]
     [29L 'Smith' 4.6]
     [23L 'Jack' 3.8]]
    

##### Head & Tail


```python
import pandas as pd
import numpy as np
 
#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

# create a DataFrame
df = pd.DataFrame(d)
print "Our object is:"
print df
print "The first two rows of the data frame is:"
print df.head(2)
print "The last two rows of the data frame is:"
print df.tail(2)
```

    Our object is:
       Age   Name  Rating
    0   25    Tom    4.23
    1   26  James    3.24
    2   25  Ricky    3.98
    3   23    Vin    2.56
    4   30  Steve    3.20
    5   29  Smith    4.60
    6   23   Jack    3.80
    The first two rows of the data frame is:
       Age   Name  Rating
    0   25    Tom    4.23
    1   26  James    3.24
    The last two rows of the data frame is:
       Age   Name  Rating
    5   29  Smith     4.6
    6   23   Jack     3.8
    


```python

```
