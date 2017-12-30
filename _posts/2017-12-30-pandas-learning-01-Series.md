
## Pandas学习笔记01
#### ——— Pandas即panel data的缩写
### pandas三种数据结构
- Series
- DataFrame
- Panel

| 数据结构 | 维度 | 描述 |
| :---- | :----: | :---- |
| Series | 1 | 一维标记大小不可变的同质数组 |
| DataFrame | 2 | 通常是二维标记的，大小可变，与潜在的异质列类型的表格结构 |
| Panel | 3 | 通常是三维标记的，大小可变的异质数组 |

### Series
#### 1. 创建一个空Series


```python
# import the pandas library and aliasing as pd
import pandas as pd
s = pd.Series()
print s
```

    Series([], dtype: float64)
    

#### 2. 从ndarray创建一个Series


```python
# import the pandas library and aliasing as pd
import pandas as pd
import numpy as np
data = np.array(['a','b','c','d'])
s = pd.Series(data)
print s
```

    0    a
    1    b
    2    c
    3    d
    dtype: object
    


```python
# another example
# import the pandas library and aliasing as pd
import pandas as pd
import numpy as np
data = np.array(['a','b','c','d'])
s = pd.Series(data,index=[100,101,102,103])
print s
```

    100    a
    101    b
    102    c
    103    d
    dtype: object
    

#### 3. 从字典创建一个Series


```python
# import the pandas library and aliasing as pd
import pandas as pd
import numpy as np
data = {'a':0.,'b':1.,'c':2.}
s = pd.Series(data)
print s
```

    a    0.0
    b    1.0
    c    2.0
    dtype: float64
    


```python
# another example
# import ...
import pandas as pd
import numpy as np
data = {'a':0.,'b':1.,'c':2.}
s = pd.Series(data,index=['b','c','d','a'])
print s
```

    b    1.0
    c    2.0
    d    NaN
    a    0.0
    dtype: float64
    

#### 4. 从标量创建一个Series


```python
# import the pandas library and aliasing as pd
import pandas as pd
import numpy as np
s = pd.Series(5,index=[0,1,2,3])
print s
```

    0    5
    1    5
    2    5
    3    5
    dtype: int64
    

#### 5. 按位从Series访问数据


```python
# retrieve the first element
import pandas as pd
s = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
print s[0]
```

    1
    


```python
# another example
import pandas as pd
s = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])

# retrieve the first three element
print s[:3],'\n'

# retrieve the last three element
print s[-3:]
```

    a    1
    b    2
    c    3
    dtype: int64 
    
    c    3
    d    4
    e    5
    dtype: int64
    

#### 6. 使用标签检索数据（索引）

一个Series就像一个固定大小的字典，你可以通过索引获取或者设置元素值


```python
# retrieve a single element
import pandas as pd
s = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
print s['a']
```

    1
    


```python
# another example
# retrieve multiple elements using a list of index label values
import pandas as pd
s = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
print s[['a','c','d']]
```

    a    1
    c    3
    d    4
    dtype: int64
    


```python
# if a label is not contained, an exception is raised
import pandas as pd
s = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
print s['f']
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-4-bacb63c57eea> in <module>()
          2 import pandas as pd
          3 s = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
    ----> 4 print s['f']
    

    C:\Users\wanghao03\AppData\Local\Continuum\Anaconda2\lib\site-packages\pandas\core\series.pyc in __getitem__(self, key)
        558     def __getitem__(self, key):
        559         try:
    --> 560             result = self.index.get_value(self, key)
        561 
        562             if not lib.isscalar(result):
    

    C:\Users\wanghao03\AppData\Local\Continuum\Anaconda2\lib\site-packages\pandas\indexes\base.pyc in get_value(self, series, key)
       1923                     raise InvalidIndexError(key)
       1924                 else:
    -> 1925                     raise e1
       1926             except Exception:  # pragma: no cover
       1927                 raise e1
    

    KeyError: 'f'



```python

```
