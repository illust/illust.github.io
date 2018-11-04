---
layout: post
title: "【论文研读】SRFeat: Single Image Super-Resolution with Feature Discrimination"
categories: [Paper Comprehension]
excerpt: 本文提出了一种新型的Simple Image Super Resolution(SISR)框架，在前人将生成对抗网络(GANs)应用到SISR领域的基础上进行了结构创新，经过实验证明了该框架达到了最前沿的(state-of-the-art)研究结果，产生了非常好的图像恢复效果。
tags: [Super-resolution, GANs]
---

## Keywords: super-resolution, adversarial network, high frequency features, perceptual quality
本篇论文的主要贡献主要包含以下两部分：
1. 提出了一个新的SISR框架，其中包括两个不同的判别器：一个工作在图像域的图像判别器，另一个是在特征域的判别器。本文的创新点正是第一次设计出了特征域判别器，能够让生成器网络产生感知上逼真的超分辨率图像。
2. 在生成器的改进上，引入了long-range skip connections，使得间隔较远的网络层之间的信息能更加高效地传播。这种设计使得生成器在PSNR指标上达到了最先进的水平。
下面是生成器网络的结构图：
![SRFeat-generator](/assets/SRFeat-generator.png)
解读：
首先在输入图像上使用9x9的卷积层提取低级特征；然后采用16个**residual block**去学习非线性程度更高、感受域更大的高级特征。这里使用的residual block结构源自SRResNet框架。
然后再利用long-range skip connection将来自不同residual blocks的特征聚集起来作为residual blocks部分最后的输出。接下来使用Sub-pixel Conv去上采样residual blocks的特征映射，以获得目标分辨率。
最后，将上采样的特征映射喂给一个3x3的卷积层得到3通道彩色图像。
