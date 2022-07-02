---
layout: post
title: "Vue3中出现的script-setup语法糖"
categories: [Vue3]
excerpt: 介绍setup语法糖。
tags: [vue3]
--- 

## Vue3的setup语法糖

Vue3.0中需要写setup函数并通过return将变量暴露出来，模板才能使用。并且需要写export default。

```<script setup>```语法的出现，摒弃了setup函数。引入的组件可以直接使用，不需要注册；name属性也不需要写了。

因为没有了setup函数，props和emit被defineProps和defineEmits所代替。defineProps用来接收父组件传来的 props。defineEmits用来由子组件向父组件传递事件。