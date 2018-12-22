---
layout: post
title: "【python】常见面试知识点收集"
categories: [python-interview]
excerpt: 持续更新... 
tags: [Python，Intern]
---
- 目录
{:toc #markdown-toc}

## 1. Python直接赋值、浅拷贝与深拷贝
- **直接赋值**：其实就是对象的引用（别名）
- **浅拷贝**：拷贝父对象，不会拷贝对象的内部的子对象
- **深拷贝**：copy模块的deepcopy方法，完全拷贝了父对象及其子对象
- **图示参见[这里](http://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html)**  
举例如下：

```python
import copy
a = [1, 2, 3, 4, ['a', 'b']] # 原始对象
 
b = a                       # 赋值，传对象的引用
c = copy.copy(a)            # 对象拷贝，浅拷贝
d = copy.deepcopy(a)        # 对象拷贝，深拷贝
 
a.append(5)                 # 修改对象a
a[4].append('c')            # 修改对象a中的['a', 'b']数组对象
 
print( 'a = ', a )
print( 'b = ', b )
print( 'c = ', c )
print( 'd = ', d )
```

输出结果如下：

```python
a =  [1, 2, 3, 4, ['a', 'b', 'c'], 5]
b =  [1, 2, 3, 4, ['a', 'b', 'c'], 5]
c =  [1, 2, 3, 4, ['a', 'b', 'c']]
d =  [1, 2, 3, 4, ['a', 'b']]

```

## 2. Python当中的可变与不可变对象：
- **可变对象**：list,dict
- **不可变对象**：int,string,float,tuple 
- **参见[这里](https://www.jianshu.com/p/c5582e23b26c)**

## 3. Python的猴子补丁
- **属性在运行时的动态替换**叫做猴子补丁
- 因为python类中的方法其实也只是一个属性，可以随时修改，所以用猴子补丁非常方便
举例如下：

```python
from SomeOtherProduct.SomeModule import SomeClass
def speak(self):
  return "I'm monkey king!"

SomeClass.speak = speak
```

## 4. Python多线程
- **进程**：对于操作系统来说，一个任务就是一个进程（Process），比如打开一个Word就启动了一个Word进程
- **线程**：在一个进程内部，要同时干多件事，就需要同时运行多个“子任务”，比如Word，它可以同时进行打字、拼写检查、打印等事情。我们把进程内的这些“子任务”称为线程（Thread）
- **多进程**：真正的并行执行多个任务只能在多核CPU上实现。但在单核CPU中，往往也能够实现多任务。操作系统的策略是轮流让各个任务交替执行，由于CPU的执行速度实在是太快了，我们感觉就像所有任务都在同时执行一样。
- **多线程**：多线程的执行方式和多进程是一样的，也是由操作系统在多个线程之间快速切换，让每个线程都短暂地交替运行，看起来就像同时执行一样。
- **多任务的实现有3种方式：**
  1.  多进程模式
  2.  多线程模式
  3.  多进程+多线程模式
- python中的多线程实现：因为Python的线程虽然是真正的线程，但解释器执行代码时，有一个GIL锁：Global Interpreter Lock，任何Python线程执行前，必须先获得GIL锁，然后，每执行100条字节码，解释器就自动释放GIL锁，让别的线程有机会执行。这个GIL全局锁实际上把所有线程的执行代码都给上了锁，所以，多线程在Python中只能交替执行，即使100个线程跑在100核CPU上，也只能用到1个核。
- Python虽然不能利用多线程实现多核任务，但可以通过多进程实现多核任务。多个Python进程有各自独立的GIL锁，互不影响。
- 参见廖雪峰[博客](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014319272686365ec7ceaeca33428c914edf8f70cca383000)

## 5. 解释Python中的help()和dir()函数
- **Help()**函数是一个内置函数，用于查看函数或模块用途的详细说明
- **Dir()**函数也是Python内置函数，dir() 函数不带参数时，返回当前范围内的变量、方法和定义的类型列表；带参数时，返回参数的属性、方法列表

## 6. Python中的字典是什么?
- 字典是C++和Java等编程语言中所没有的数据类型，它具有键值对
- 字典是不可变的，我们也能用一个推导式来创建它
举例如下：

```python
roots = {x**2:x for x in range(5,0,-1)}
roots
# 运行结果
{25: 5, 16: 4, 9: 3, 4: 2, 1: 1}
```

## 7. 请解释使用*args和**kwargs的含义
- *args和**kwargs主要用于函数定义，便于将不定量的参数传递给函数
- 写成*args和**kwargs形式只是一个通俗的命名约定，只有前面的星号是必须的
- *args 是用来发送一个非键值对的可变数量的参数列表给一个函数
- **kwargs 允许你将不定长度的键值对, 作为参数传递给一个函数。 如果你想要在一个函数里处理带名字的参数, 你应该使用**kwargs
- 参见[这里](https://eastlakeside.gitbooks.io/interpy-zh/content/args_kwargs/)

## 8. 解释Python中的join()和split()函数
- **Join()**能让我们将指定字符添加至字符串中
- **Split()**能让我们用指定字符分割字符串

## 9. Python中的闭包是什么？
在一个嵌套函数的内部函数中，对外部作用域的变量进行引用，(并且一般外部函数的返回值为内部函数)，那么内部函数就被认为是闭包。
- 注意：**闭包无法修改外部函数的局部变量**。


## 10. 在python中有多少种运算符？
在Python中，我们有7种运算符：算术运算符、关系运算符、赋值运算符、逻辑运算符、位运算符、成员运算符、身份运算符。

## 11. staticmethod和classmethod的用法和区别：
- Python中的类也是一个普通对象，如果需要直接使用这个类，例如将类作为参数传递到其他函数中，又希望在实例化这个类之前就能提供某些功能，那么最简单的办法就是使用classmethod和staticmethod
- 这两者的**区别**在于在存在类的继承的情况下**对多态的支持不同**
- 具体解释参见[Python 中的 classmethod 和 staticmethod 有什么具体用途？ - 灵剑的回答 - 知乎](https://www.zhihu.com/question/20021164/answer/537385841)

## 12. Python中的描述符
- 参见[Python描述符入门指北](https://manjusaka.itscoder.com/2016/10/12/Something-about-Descriptor/)

## 13. Python2.7.x和python3.x的主要差异  
- __future__模块
- print函数
- 整除
- Unicode
- xrange模块
- 详见[Python 2.7.x 与 Python 3.x 的主要差异](http://chenqx.github.io/2014/11/10/Key-differences-between-Python-2-7-x-and-Python-3-x/)

## 14. Python中的is和"=="
- is判断地址是否相同，"=="判断值是否相同  


