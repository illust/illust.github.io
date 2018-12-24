---
layout: post
title: "【最优化】Half Quadratic Splitting"
categories: [Optimization]
excerpt: 持续更新... 
tags: [优化算法]
---
## Half Quadratic Splitting（HQS,半二次方分裂算法）  
因为接触[Learning Deep CNN Denoiser Prior for Image Restoration](https://arxiv.org/abs/1704.03264)这篇论文，当中提到的两种变量分离技术ADMM(alternating direction method of multipliers,交替方向乘子法)和HQS。我看到网络上关于ADMM的介绍还是挺多的，比如知乎上的这篇答案：[交替方向乘子法（ADMM）算法的流程和原理是怎样的？](https://www.zhihu.com/question/36566112)，有一个科大女神的回答挺不错的。  
但是关于HQS的中文总结就寥寥无几了，但是也能找到一篇在对这个论文的解读的[博客](http://happycaoyue.com/2018/04/07/Learning%20Deep%20CNN%20Denoiser%20Prior%20for%20Image%20Restoration/)，当中简单翻译了英文原文。  

