---
layout: post
title: numpy中argsort()函数用法
categories: [blog]
excerpt: 今天我开通博客了！
---

## numpy中argsort()的使用
***
argsort()是用于numpy数组中的元素排序的一种函数,它的返回值是数组中元素排序后的原下标。
例如，np.argsort([2,1,3])的返回结果是array([1,0,2],dtype=int64),结果数组中的元素1就表示对应元素1的起始下标，以此类推。
使用argsort()的优势就是可以不改变起始数组的元素顺序，按照下标索引的方法对其进行排序。
***


```python
>> import numpy as np

# 随机生成[0,100)内的整数，步长为10
>> x = np.random.randint(0,100,10)
>> x
array([56, 13,  3, 64, 53, 37, 33,  3, 76, 17])

# 返回元素原下标数组
>> np.argsort(x)
array([2, 7, 1, 9, 6, 5, 4, 0, 3, 8], dtype=int64)

# 利用下标索引对数组x进行升序排序
>> x[np.argsort(x)]
array([ 3,  3, 13, 17, 33, 37, 53, 56, 64, 76])

# 扩展一下，按数组原来的顺序返回最大的5个数
# sorted()用于对下标数组进行升序排序
>> x[sorted(np.argsort(x)[-5:])]
array([56, 64, 53, 37, 76])
```

### 关于我
你可以直接下载[我的简历]({{ site.url }}/assets/SvenWong-ResumeCN.pdf).

