---
layout: post
title: "【学习OpenCV3】滤波与卷积"
categories: [OpenCV]
excerpt: 通过OpenCV提供的API学习图像的滤波操作
tags: [OpenCV,DIP]
---
- 目录
{:toc #markdown-toc}


# 1. 预备知识

## 1.1 滤波、核和卷积
概念：
**线性核**的**滤波**就是我们所说的**卷积**操作，只是通常前者是信号处理的术语，后者是计算机视觉的术语。
**核的锚点**定义了核与源图像的对齐关系。

## 1.2 边界外推和边界处理
在使用OpenCV进行边缘处理的时候，我们需要知道，OpenCV中的滤波操作（如cv::blur()，cv::erode()，cv::dilate()等）得到的输出图像与源图像的形状大小是一致的。OpenCV内部是采用对源图像周围添加虚拟像素的操作，从而达到这种效果。
这里主要涉及到两个函数：自定义边框（cv::copyMakeBorder()）和自定义外推（cv::borderInterpolate()），下面给出具体介绍。

- cv::copyMakeBorder()通过指定两幅图像，同时指明填充方法，该函数就会将第一幅图像填充后的结果保存在第二幅图像中。第三到第六个参数表示边框四个方向的尺寸；边框类型参数详见[cv::borderType](https://docs.opencv.org/3.2.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5)。

~~~cpp
void cv::copyMakeBorder(
  cv::InputArray    src,                      // Input image
  cv::OutputArray   dst,                      // Result image
  int               top,                      // Top side padding (pixels)
  int               bottom,                   // Bottom side padding (pixels)
  int               left,                     // Left side padding (pixels)
  int               right,                    // Right side padding (pixels)
  int               borderType,               // Pixel extrapolation method
  const cv::Scalar& value = cv::Scalar()      // Used for constant borders
);
~~~

- cv::borderInterpolate()给定边框虚拟像素，计算对应的源图像中的参考像素位置。参数1是指定维度上的坐标，参数2是指定维度的大小，参数3是边框类型。

~~~cpp
int cv::borderInterpolate(    // Returns coordinate of "donor" pixel
  int p,                      // 0-based coordinate of extrapolated pixel
  int len,                    // Length of array (on relevant axis)
  int borderType              // Pixel extrapolation method
);
~~~

这两个函数在OpenCV内部经常使用，用于支持滤波操作。

# 2. 阈值化操作
图像处理过程中经常需要在完成多层处理步骤以后做出一个最终决定，或者将高于或低于某一值的像素置零的同时其他像素保持不变。cv::threshold()实现了这些功能，它**对输入单通道矩阵逐像素进行固定阈值分割**。典型应用是从灰度图像获取二值图像，或消除灰度值过大或过小的噪声。有5种阈值分割类型，由参数thresholdType决定，详见[cv::ThresholdTypes](https://docs.opencv.org/3.2.0/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576)。

~~~cpp
double cv::threshold(
  cv::InputArray  src,           // Input image
  cv::OutputArray dst,           // Result image
  double          thresh,        // Threshold value
  double          maxValue,      // Max value for upward operations
  int             thresholdType  // Threshold type to use
);
~~~

## 2.1 Otsu算法
cv::threshold()也可以自动决定最优的阈值，只需要对参数thresh传递值cv::THRESH_OTSU。
Otsu算法也称为大津算法，运用的是聚类的原理，即：类内方差最小，类间方差最大。参见[图像二值化与otsu算法介绍](https://www.jianshu.com/p/c7fb9be02412)

## 2.2 自适应阈值
还有一种阈值化方法cv::adaptiveThreshold()，这种方法阈值在整个过程中自动产生变化。具体操作是该函数根据adaptiveMethod的设置，允许两种不同的自适应阈值方法。两种方法都是逐个像素地计算自适应阈值T(x,y)，方法是通过计算每个像素位置周围的bxb区域的加权平均值然后减去常数C，其中b由blockSize给定。

~~~cpp
void cv::adaptiveThreshold(
  cv::InputArray  src,            // Input image
  cv::OutputArray dst,            // Result image
  double          maxValue,       // Max value for upward operations
  int             adaptiveMethod, // Mean or Gaussian
  int             thresholdType,  // Threshold type to use
  int             blockSize,      // Block size
  double          C               // Contant
~~~

相对于一般的阈值化操作，当图像中出现较大的明暗差异时，自适应阈值的效果更好。

# 3. 平滑

>平滑滤波是低频增强的空间域滤波技术。它的目的有两类：一类是**模糊**；另一类是**消除噪音**。空间域的平滑滤波一般采用简单平均法进行，就是求邻近像元点的平均亮度值。邻域的大小与平滑的效果直接相关，邻域越大平滑的效果越好，但邻域过大，平滑会使边缘信息损失的越大，从而使输出的图像变得模糊，因此需合理选择邻域的大小。
作者：Darlingqiang
原文地址：[图像平滑处理，6种滤波总结；](https://blog.csdn.net/Darlingqiang/article/details/79507468)

下面分别介绍OpenCV提供的5种不同的平滑操作，每种结果有着细微的区别。

## 3.1 简单模糊和方框滤波器
- cv::blur()实现了简单模糊，目标图像中的每个值都是源图像中相应位置一个窗口（核）中像素的平均值。

~~~cpp
void cv::blur(
  cv::InputArray  src,                             // Input image
  cv::OutputArray dst,                             // Output image
  cv::Size        ksize,                           // Kernel size
  cv::Point       anchor      = cv::Point(-1,-1),  // Location of anchor point
  int             borderType  = cv::BORDER_DEFAULT // Border extrapolation to use
);
~~~

- 方框型滤波器cv::boxFilter()是一种矩形的并且滤波器中所有值全部相等的滤波器。其中，所有值为1/A（A为滤波器面积）的滤波器称为“归一化方框型滤波器”，也即均值滤波器。

~~~cpp
void cv::boxFilter(
  cv::Input  src,                               // Input image
  cv::Output dst,                               // Result image
  int        ddepth,                            // Output depth (e.g., CV_8U)
  cv::Size   ksize,                             // Kernel size
  cv::Point  anchor    = cv::Point(-1,-1)       // Location of anchor point
  bool       normalize = true,                  // If true, divide by box area
  int        borderType  = cv::BORDER_DEFAULT   // Border extrapolation to use
);
~~~

cv::boxFilter()是一种一般化的形式，而cv::blur()是一种特殊化的形式。两者根本区别主要是前者可以以非归一化形式调用（normalize=true时，为均值滤波；normalize=false时，为方框滤波），并且输出图像深度可以控制。另外，方框型滤波器会让图像的边缘信息丢失。

## 3.2 中值滤波器
cv::medianBlur()将每个像素替换为围绕这个像素的矩形邻域内的中值或“中值”像素。**与均值滤波相比较，少量具有较大偏差的点会严重影响到均值滤波，而中值滤波采用取中间点的方式来消除异常值。**

~~~cpp
void cv::medianBlur(
  cv::InputArray  src,        // Input image
  cv::OutputArray dst,        // Result image
  cv::Size        ksize       // Kernel size
);
~~~

## 3.3 高斯滤波器
高斯函数具有五个重要的性质，这些性质使得它在早期图像处理中特别有用．这些性质表明，高斯平滑滤波器无论在空间域还是在频率域都是十分有效的低通滤波器，且在实际图像处理中得到了工程人员的有效使用。五个性质列举如下：

- 二维高斯函数具有旋转对称性，即滤波器在各个方向上的平滑程度是相同的；
- 高斯函数是单值函数；
- 高斯函数的傅立叶变换频谱是单瓣的；
- 高斯滤波器宽度(决定着平滑程度)是由参数σ表征的，而且σ和平滑程度的关系是非常简单的；
- 由于高斯函数的可分离性，较大尺寸的高斯滤波器可以得以有效地实现。

详见[图像滤波之高斯滤波介绍](http://imgtec.eetrend.com/d6-imgtec/blog/2018-04/11426.html)

~~~cpp
void cv::GaussianBlur(
  cv::InputArray  src,                             // Input image
  cv::OutputArray dst,                             // Output image
  cv::Size        ksize,                           // Kernel size
  double          sigmaX,                          // Gaussian half-width in x-direction
  double          sigmaY   = 0.0,                  // Gaussian half-width in y-direction
  int             borderType = cv::BORDER_DEFAULT  // Border extrapolation to use
);
~~~

## 3.4 双边滤波器
双边滤波器是一种比较大的图像分析算子，也就是边缘保持平滑。

双边滤波的效果就是将源图像变成一幅水彩画，这种效果在多次迭代后更加明显，因此这种方法在**图像分割**领域十分有用。


~~~cpp
void cv::bilateralFilter(
  cv::InputArray  src,                             // Input image
  cv::OutputArray dst,                             // Result image
  int             d,                               // Pixel neighborhood size (max distance)
  double          sigmaColor,                      // Width param for color weight function
  double          sigmaSpace,                      // Width param for spatial weight function
  int             borderType = cv::BORDER_DEFAULT  // Border extrapolation to use
~~~

# 4. 导数和梯度

## 4.1 索贝尔导数

## 4.2 Scharr滤波器

## 4.3 拉普拉斯变换


# 5. 图像形态学

## 5.1 膨胀和腐蚀

## 5.2 通用形态学函数

## 5.3 开操作和闭操作

## 5.4 形态学梯度

## 5.5 自定义核


# 6. 用任意线性滤波器做卷积

## 6.1 用cv::filter2D()进行卷积

## 6.2 通过cv::sepFilter2D使用可分核

## 6.3 生成卷积核

