---
title: 神经网络量化
date: 2019-11-30 15:23:39
tags:
  - algorithm
  - quantization
categories: algorithm
---

> 量化是指将连续分布的数据或者较大的数据集合转化为离散形式的处理。在信号处理、图片处理、数据压缩等方面，量化都有着诸多的应用，而本文将聚焦于神经网络中的量化。

<!--more -->

## 背景

现有的大多深度学习应用都是基于32位的浮点进行训练或者推断的。现有一些方法尝试使用低精度的表示完成运算，比如训练任务采用16bit乘法器，或者是在推断任务中使用8bit乘法器。数据从高精度表示到低精度表示的转换我们统称为量化（Quantization）。随着深度学习方法在业界的逐步落地，量化也逐渐成为算法落地过程中必经的步骤之一。

低精度表示带来的好处主要有两方面，其一是更低的内存带宽占用，实现快速的运算需要有高速的内存，而内存的存取速度与其价格成正比，所以内存大小能直接影响到成本；其二是更大的计算吞吐量，该指标一般使用每秒运算数（Operation per second，简称OPS）衡量，低精度运算所需要的芯片面积可以更小，例如Intel等芯片厂商已经开发了将4个INT8数据拼接为32bit进行运算的相关指令，这样同一芯片执行力INT8运算的OPS等价提升为32bit运算的四倍，非常可观。

量化一般分为两种，分别是训练后量化（Post-Training Quantization）和训练量化。前者是基于充分训练的浮点模型，将其权重或者激活值转化为低比特数表示。在转化之后，部分模型需要finetune实现更低的精度损失。而训练量化是基于一个未经充分训练（甚至是随机初始化）的模型，将其权重、激活值和梯度的部分或者全部转化为低比特表示，利用低比特表示进行训练，使得该模型的效果逼近浮点模型的效果。上述的分类方法只是从是否需要训练的角度进行划分，当然也可以从其他的角度进行分类，比如根据低比特表示的分布是否均匀可分为均匀量化和非均匀量化；根据低比特表示覆盖范围是否涵盖原始数据的范围可以分为饱和量化和非饱和量化。更细节的内容将在下面的章节介绍。

## 历史

现有的量化算法基本可以实现基于`16bit`乘法器的无损训练，或者是基于`8bit`乘法器的无损推断。

## 基本量化方案

现有的表示方案主要有以下三类：均匀量化（Uniform Quantization）、2的指数量化（Power of 2 Quantization）和浮点量化（Float Quantization）。

### 均匀量化

均匀量化指的是通过一个 `scale` 参数，将原始数据 `x` 转化为该参数和低比特表示 `x'` （一般为整形值）的乘积，即 `scale*x'` 。

均匀量化一般有以下三个步骤：

1. 搜索数据的极值
2. 根据极值计算量化参数
3. 量化转换

<!-- ```mermaid
graph TB;
    搜索数据的极值
    根据极值计算量化参数;
    量化转换;
``` -->

对于均匀量化，我们采用如下形式化表示。假设有一组分布在区间$[x_{min}, x_{max}]$的浮点数值需要量化到无符号的$(0,2^{b}-1)$范围内，其中$b$表示量化表示的比特数，比如对于 `8bit` 量化共有$2^8$个数值。将浮点值转化为整形量化表示的过程需要有两个参数：`scale` 和 `zero-point`（$z$） ，`scale` 表示量化的步长，是一种范围的伸缩变换；而 `zero-point` 则是一种范围的位移变换。`zero-point` 是一个整形值，用于确保0值的量化是没有误差的。

基于上述的问题，`scale` 和 `z` 的计算方式分别如下
$$
scale = \frac{x_{max} - x_{min}}{2^{b}-1}
$$
$$
z = -round(\frac{x_{min}}{scale})
$$
一旦参数 `scale` 和 `zero-point`（$z$） 得以确定，量化就可以通过下面的方式进行：
$$
x' = round(\frac{x}{scale})+z
$$
$$
x_{int} = clamp(x', 0, 2^{b}-1)
$$
其中，$clamp(x, a, b)=min(max(a, x), b)$。而反量化就是上述的逆过程：
$$
x_{rec} = (x_{int} - z) \times scale
$$

#### 对称与非对称量化

关于对称量化和非对称量化的区别，一种说法是通过量化的区间是否关于`0`对称来判断，还有一种说法是当$z=0$即是对称量化，反之则不是。姑且不论二者的正统与否，毕竟其内核是一致的。对称量化的操作一般如下：

$$
x' = round(\frac{x}{scale})
$$
$$
x_{int} = clamp(x', -2^{b-1}+1, 2^{b-1}-1), 当x是有符号数
$$
$$
x_{int} = clamp(x', 0, 2^b-1), 当x是无符号数
$$

对ReLU层输出的激活值可以做无符号的量化，而其他层激活值则一般只能是有符号的量化。

#### 浮点步长与定点量化

当 `scale` 参数是一个浮点数，此时的均匀量化可以称之为浮点步长均匀量化。当 `scale` 参数是 $2^n$ 这种表示时，该量化形式称之为定点量化。通过上述的方法所计算出的 `scale` 一般是一个浮点数，即对应了浮点步长量化。所以接下来介绍一种带符号位的动态定点的量化方法。

在介绍定点量化的流程之前，我们先来看下定点的表示方案。定点是将数据使用整形值 $x_{int}$ 和一个共享指数 $e$ 联合表示，即
$$
x_{rec}=x_{int} \times 2^{e}
$$
定点量化的关键就是要找到合适的共享指数 $e$，使得总体的量化损失最小。一般地，我们认为大值是更为关键的，对神经网络的输出有着直接的影响，所以要保证大值尽可能地被覆盖。而小值的量化结果可能就是$x_{int}=0$，也就是以0替代。

根据上述基本策略，定点量化步骤如下

<!-- ```mermaid
graph TB;
    Step1:从原始数据搜索最大值
    Step2:计算最大值的指数部分;
    Step3:根据最大值指数计算数据的共享指数;
    Step4:数据变换;
``` -->

<!-- 上述个步骤对应的数学表达如下， -->

Step1，从原始数据搜索最大值：
$$
f=max(|x|)
$$
Step2，计算最大值的指数部分：
$$
e_{f}=floor(log2(f)) + 1
$$
Step3，根据最大值指数计算数据的共享指数：
$$
e=e_{f} - (b-1)
$$
Step4，数据变换：
$$
x_{int}=round(x/2^e)
$$
其中，$b$表示整形值部分的比特数，而$b-1$则是去掉符号位一个比特之后的比特数。一般地，还需要对量化的结果约束在$b$个比特可以表示的范围内，即$clamp(x_{int}, -2^{b-1}-1, 2^{b-1}-1)$。

### 浮点量化

浮点量化就是采用低精度的浮点格式替换高精度的浮点数。在解释浮点量化之前，首先介绍下浮点数的格式。浮点是相对于定点而言的，即其小数点的位置是不固定的。根据IEEE 754标准，浮点数的表示可以分为三个部分，分别是：符号位（sign）、尾数（mantissa）和指数（exponent）。这三个部分的解释如下：

- 符号位（记做$S$）由一个比特表示，实际表示的值为$(-1)^S$，当$S=0$表示该浮点数为正数；当$S=1$表示该浮点数为负数

- 尾数部分（记做$M$）有若干比特，首位含有一个隐藏的比特，实际表示的值是介于1到2之间的小数

- 指数部分（记做$E$）有若干比特，实际表示的值为$2^E$

浮点数的所存储的值就是三者的乘积，即
$$
Float\_value = (-1)^S \times M \times 2^{E}
$$

IEEE 754标准还规定的了现今两种使用最多的浮点数格式，分别是32比特的单精度浮点（Single precision）和64比特的双精度浮点（Double precision）。单精度浮点的32比特划分为（1，8，23）三个部分，分别是符号位、指数部分和尾数部分的比特数。

![avatar][singlefloatformat]

尾数部分按照如下的方式解析。该部分的第一个比特表示$2^{-1}$，第二个比特表示$2^{-2}$，以此类推。另外尾数部分隐含了$1.0$，所以尾数部分的真实值是 $1.0 + m_{1} \times 2^{-1} + m_{2} \times 2^{-2} + ...$，其中$m_{i}$表示尾数部分第 $i$ 个比特的值。

指数部分的所有比特构成一个无符号的整形数（记做$U$），比如单精度浮点的指数部分有8个比特，那么该无符号整形数的取值范围为 $U\in[0,255]$。为了同时可以兼顾大值和小值的表示，该无符号整形和真实的$E$之间存在一个偏置。对于单精度浮点，该偏置的值为`127`。直白地说，$E=U-127$，因此有$E\in[-127,128]$。尽管理论上$E$是该取值范围，但按照刚才的方法，$(-1)^S \times M \times 2^{E}$是没有办法准确表示 `0` 的。实际上$E$的解析还存在两种特殊情况：

- 指数$E$所有比特全为0。此时，浮点数指数的$E$等于 `1-127`，尾数部分$M$不再包含隐含的$1.0$。这样就可以表示 $\pm 0$和一些很小的数。

- 指数$E$所有比特全为1。此时，如果尾数$M$所有比特全为0，表示无穷大$\pm \inf$；如果尾数$M$不全为0，表示该数不是一个数（即NaN）。

双精度浮点共64比特，含1个比特的符号位、11个比特的指数和52个比特的尾数，指数部分的偏置为1023，除此之外双精度浮点和单精度是一致的。

上文发费大量篇幅介绍浮点数的表示方法并非无用功，只有对浮点数的表示足够熟悉才能准确高效地实现浮点量化。当前大多深度学习算法都是基于单精度浮点数进行训练和推理的，而浮点量化就是用低比特的浮点数替代单精度浮点。常见的低比特浮点数有FP16和FP8，前者是16比特的浮点，后者是8比特的浮点。每种浮点数又可以定义不同的格式，比如FP16可以选择5比特指数部分和10比特尾数部分，也可以选择6比特指数部分和9比特尾数部分。不同的格式选择实际上体现了范围和精度之间的权衡。目前基于FP16的训练已经比较成熟，该方案最早由英伟达和百度在“[混合精度训练](https://arxiv.org/abs/1710.03740)”一文中提出。该文章在权重、激活值和梯度均采用了FP16格式（具体来说是IEEE 754-2008中定义的半精度浮点格式），所实现的精度基本达到或超过单精度浮点模型的基准。

<!-- ### 2的指数量化

待续。 -->

<!-- ## 进阶方法

### KL散度最小化

> 上文介绍量化方案的时候所描述的都是饱和量化的思路，而KL散度最小化量化是一种非饱和量化。二者的区别主要在于饱和量化会覆盖数据分布的整个范围；而非饱和量化则会寻求一个阈值，并将阈值之外的值以阈值替代，这样所需量化的范围得以大大减小。

[KL散度最小化量化](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)由英伟达提出，该方法旨在通过KL散度搜索得到量化的阈值。

### Clip_ReLU

待续。 -->

## 卷积神经网络量化

本节主要对卷积神经网络中权重、激活值和梯度三个部分的量化展开介绍。待续。

### 权重的量化

### 激活值的量化

### 梯度的量化

<!-- STE（Straight through estimator）梯度估计。 -->

## 参考文献

[Ristretto: Hardward-Oriented Approximation of Convolution Neural Networks](https://arxiv.org/abs/1605.06402)

[Lower Numerical Precision Deep Learning Inference and Training](https://software.intel.com/sites/default/files/managed/db/92/Lower-Numerical-Precision-Deep-Learning-Jan2018.pdf)

[Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/abs/1806.08342)

[singlefloatformat]:data:image/png;base64,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