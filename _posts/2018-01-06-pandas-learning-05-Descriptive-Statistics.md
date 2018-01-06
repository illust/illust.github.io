
## 描述性统计

### 1. sum(),mean(),std(),...


```python
import pandas as pd
import numpy as np

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])}

# create a DataFrame
df = pd.DataFrame(d)
print df.sum() # by default,axis is index(axis=0)
print df.sum(1),'\n'
print df.mean(),'\n'
print df.std()
```

    Age                                                     382
    Name      TomJamesRickyVinSteveSmithJackLeeDavidGasperBe...
    Rating                                                44.92
    dtype: object
    0     29.23
    1     29.24
    2     28.98
    3     25.56
    4     33.20
    5     33.60
    6     26.80
    7     37.78
    8     42.98
    9     34.80
    10    55.10
    11    49.65
    dtype: float64 
    
    Age       31.833333
    Rating     3.743333
    dtype: float64 
    
    Age       9.232682
    Rating    0.661628
    dtype: float64
    

### 2. 函数及描述汇总

| No. | 函数 | 描述 |
| :---- | :----: | :---- |
| 1 | count() | 非空观测值的数量 |
| 2 | sum() | 值的总和 |
| 3 | mean() | 值的均值 |
| 4 | median() | 值的中位数 |
| 5 | mode() | 值的取模 |
| 6 | std() | 值的标准差 |
| 7 | min() | 最小值 |
| 8 | max() | 最大值 |
| 9 | abs() | 绝对值 |
| 10 | prod() | 值的乘积 |
| 11 | cumsum() | 累积和 |
| 12 | cumprod() | 累计乘 |

### 总结性数据describe()
##### describe()函数计算与DataFrame列有关的统计信息的摘要


```python
import pandas as pd
import numpy as np

# create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
                      'Lee','David','Gasper','Betina','Andres']),
    'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])}

# create a DataFrame
df = pd.DataFrame(d)
print df,'\n'
print df.describe()
```

        Age    Name  Rating
    0    25     Tom    4.23
    1    26   James    3.24
    2    25   Ricky    3.98
    3    23     Vin    2.56
    4    30   Steve    3.20
    5    29   Smith    4.60
    6    23    Jack    3.80
    7    34     Lee    3.78
    8    40   David    2.98
    9    30  Gasper    4.80
    10   51  Betina    4.10
    11   46  Andres    3.65 
    
                 Age     Rating
    count  12.000000  12.000000
    mean   31.833333   3.743333
    std     9.232682   0.661628
    min    23.000000   2.560000
    25%    25.000000   3.230000
    50%    29.500000   3.790000
    75%    35.500000   4.132500
    max    51.000000   4.800000
    


```python

```
