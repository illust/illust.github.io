---
layout: post
title: "【OpenCV】常用数组操作整理"
categories: [opencvlearning]
excerpt: 回到写日记的年月里，fight！
tags: [opencv]
---
# cv::normalize函数
**两种声明形式：**
第一种：

~~~ cpp
void cv::normalize(
  cv::InputArray src,
  cv::OutputArray dst,
  double alpha = 1,
  double beta = 0,
  int normType = cv::NORM_L2,
  int dtype = -1,
  cv::InputArray mask = cv::noArray()
);
~~~

第二种：

~~~ cpp
void cv::normalize(
  cv::InputArray src,
  cv::SpareMat& dst,
  double alpha = 1,
  int normType = cv::NORM_L2,
);
~~~
