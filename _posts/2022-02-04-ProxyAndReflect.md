---
layout: post
title: "代理与反射"
categories: [javascript]
excerpt: 概念的简单介绍。
tags: [javascript]
--- 
---  
- 目录
{:toc #markdown-toc}

## **1 代理基础**
在ES6中，新增的代理和反射为开发者提供了拦截并向基本操作嵌入额外行为的能力。具体来说，可以给目标对象定义一个关联的代理对象，而这个代理对象可以作为抽象的目标对象来使用。在对目标对象的各种操作影响目标对象之前，可以在代理对象中对这些操作加以控制。
### **1.1 创建空代理**
创建空代理即在代理对象上执行的所有操作都会无障碍地传播到目标对象。
代理是使用Proxy构造函数创建的。这个构造函数接受两个参数：目标对象和处理程序对象。
代码示例如下：
```javascript
const target = {
    id: 'target'
};

const hander = {};

const proxy = new Proxy(target,handler);
```
