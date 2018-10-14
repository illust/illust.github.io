---
layout: post
title: Pandas学习笔记03
categories: [blog]
excerpt: Hello World!
tags: [free time]
---

## pandas学习笔记03

### pandas.panel结构
pandas.Panel(data,items,major_axis,minor_axis,dtype,copy)

#### 1. 创建Panel
- 从ndarrays创建
- 从DataFrames字典创建

##### 从3维ndarray创建


```python
# creating an empty panel\
import pandas as pd
import numpy as np

data = np.random.rand(2,4,5)
p = pd.Panel(data)
print p
```

    <class 'pandas.core.panel.Panel'>
    Dimensions: 2 (items) x 4 (major_axis) x 5 (minor_axis)
    Items axis: 0 to 1
    Major_axis axis: 0 to 3
    Minor_axis axis: 0 to 4


##### 从DataFrame对象字典创建


```python
# creating an empty panel
import pandas as pd
import numpy as np
data = {'Item1':pd.DataFrame(np.random.randn(4,3)),
       'Item2':pd.DataFrame(np.random.randn(4,2))}
p = pd.Panel(data)
print p
```

    <class 'pandas.core.panel.Panel'>
    Dimensions: 2 (items) x 4 (major_axis) x 3 (minor_axis)
    Items axis: Item1 to Item2
    Major_axis axis: 0 to 3
    Minor_axis axis: 0 to 2


#### 创建一个空Panel


```python
# creating an empty panel
import pandas as pd
p = pd.Panel()
print p
```

    <class 'pandas.core.panel.Panel'>
    Dimensions: 0 (items) x 0 (major_axis) x 0 (minor_axis)
    Items axis: None
    Major_axis axis: None
    Minor_axis axis: None


#### 2. 从Panel选择数据

从panel选择数据使用：
- Items
- Major_axis
- Minor_axis

##### 使用Items


```python
# creating an empty panel
import pandas as pd
import numpy as np
data = {'Item1':pd.DataFrame(np.random.randn(4,3)),
       'Item2':pd.DataFrame(np.random.randn(4,2))}
p = pd.Panel(data)
print p['Item1']
```

              0         1         2
    0  0.722992 -1.067307 -0.067804
    1 -0.663662  0.772226  0.082438
    2 -0.426947 -0.036010 -0.494841
    3  2.124745 -0.172040  1.030734


##### 使用major_axis, minor_axis


```python
import pandas as pd
import numpy as np
data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)),
        'Item2' : pd.DataFrame(np.random.randn(4, 2))}
p = pd.Panel(data)
print p['Item1'],'\n',p['Item2'],'\n'
print p.major_xs(1),'\n' # 1 indicated row index
print p.minor_xs(2) # 2 indicated col index
```

              0         1         2
    0  0.267236  0.511752 -0.976870
    1  0.068260  0.586807  0.268806
    2  1.154739  1.528244  1.124225
    3 -0.173479  0.281171 -1.515341
              0         1   2
    0  0.092332 -0.592215 NaN
    1 -2.058858 -0.896308 NaN
    2 -0.586212  0.389448 NaN
    3  2.188792  1.663543 NaN

          Item1     Item2
    0  0.068260 -2.058858
    1  0.586807 -0.896308
    2  0.268806       NaN

          Item1  Item2
    0 -0.976870    NaN
    1  0.268806    NaN
    2  1.124225    NaN
    3 -1.515341    NaN



```python

```
