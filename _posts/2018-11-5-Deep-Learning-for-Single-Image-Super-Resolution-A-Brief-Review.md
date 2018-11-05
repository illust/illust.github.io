---
layout: post
title: "【论文研读】Deep Learning for Single Image Super Resolution: A Brief Review"
categories: [Paper Comprehension]
excerpt: 本文是对基于深度学习的图像超分辨率的一篇近期回顾性论文的详细研读。
tags: [deep learning, super resolution]
---

"Deep Learning for Single Image Super Resolution: A Brief Review"这篇文章可以在[这里](https://arxiv.org/abs/1808.03344)获得。
从标题可以看出，文章是对近期（2014～2018年）深度学习在单幅图像超分辨率(SISR)领域的一些具有代表性意义的算法模型的梳理。
众所周知，单幅图像超分辨率是一个不适定(ill-posed)的挑战性问题，这是由于指定的低分辨率输入图像通常能够和很多个潜在高分辨率原图像形成对应关系，并且超分任务的特殊性往往使得我们不能获取到映射LR图像的自然图像空间。非DL的SISR解决方法通常有两个缺点：1.LR空间和HR空间之间的映射通常没有明确定义；2.在给定大量原始图像数据的情况下，建立复杂高维映射的低效性。而基于DL的SISR方法在由于数量上和质量上的显著提升，迅速成为研究的热点。
文章的主体内容划分为两大部分：1.用于SISR的有效的神经网络结构的探究；2.高效的优化目标的研究。作者提到这是由于在SISR任务中，DL和领域知识的结合是成功的关键，并且通常反映在这两个方面的创新点上。
在讨论网络结构的第一部分，以SRCNN作为benchmark，针对提出的三点可以改进的问题，分别讨论相关的每个网络结构的特点。这三个问题分别是：
1.    SRCNN的输入是LR的双三次插值，它是HR图像的近似。然而插值输入有三点缺陷：(a)输入带来的细节平滑影响可能导致进一步对图像结构的错误评估；(b)插值很耗时；(c)当下采样内核未知时，作为原始估计的一个特定内插输入是不合理的。(？)所以第一个问题就是能不能设计CNN结构直接将LR作为输入处理这些问题？
2.    SRCNN网络深度只有三层，是不是设计出更复杂（深度，宽度以及拓扑结构）的CNN框架就能够实现更好的结果？如果是，怎样去设计？
3.    我们是够可以将SISR过程的任何属性集成到CNN框架的设计或者SISR算法中的其他部分中，去更有效地处理问题？






论文中涉及到的应用到SISR的DL神经网络模型以及出处列举如下：
1. **SRCNN** [Learning a deep convolutional network for image super-resolution](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
2. **FSRCNN** [Accelerating the super-resolution
convolutional neural network](https://arxiv.org/abs/1608.00367)
3. **ESPCN** [Real-time single image and video superresolution using an efficient sub-pixel convolutional neural network](https://arxiv.org/abs/1609.05158)
4. **VDSR** [ccurate image super-resolution
using very deep convolutional networks](https://arxiv.org/abs/1511.04587)
5. **DRCN** [Deeply-recursive convolutional
network for image super-resolution](https://arxiv.org/abs/1511.04491)
6. **SRResNet** [Photo-realistic single
image super-resolution using a generative adversarial network](https://arxiv.org/abs/1609.04802)
7. **EDSR, MDSR** [Enhanced deep
residual networks for single image super-resolution](https://arxiv.org/abs/1707.02921)
8. **SRDenseNet** [Image super-resolution using
dense skip connections](http://ieeexplore.ieee.org/document/8237776/)
9. **Memnet** [Memnet: A persistent memory
network for image restoration](https://arxiv.org/abs/1708.02209)
10. **RDN** [Residual dense
network for image super-resolution](https://arxiv.org/abs/1802.08797)
11. **SCN, CSCN** [Deep networks for
image super-resolution with sparse prior](https://ieeexplore.ieee.org/document/7410407/),[Robust
single image super-resolution via deep networks with sparse prior](https://ieeexplore.ieee.org/iel7/83/4358840/07466062.pdf)
12. **DEGREE** [Deep
edge guided recurrent residual learning for image super-resolution](https://arxiv.org/abs/1604.08671)
13. **LapSRN** [eep laplacian
pyramid networks for fast and accurate super-resolution](https://arxiv.org/abs/1704.03915)
14. **DBPN** [Deep backprojection
networks for super-resolution](https://arxiv.org/abs/1803.02735)
15. **PixelRNN, PixelCNN** [Pixel recurrent
neural networks](https://arxiv.org/abs/1601.06759), [Conditional image generation with pixelcnn decoder](https://arxiv.org/abs/1606.05328)
16. **ZSSR** [Zero-shot super-resolution using
deep internal learning](https://arxiv.org/abs/1712.06087)

