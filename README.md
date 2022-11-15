# **DeflickerCycleGAN v1.0**
An offcial implement of "DeflickerCycleGAN : learning to detect and remove flickers in a single image" with [tensorflow](https://www.tensorflow.org/).
It is an interesting and practical framework for eleminate the flickers in images. The details can be found in the paper, which is submitted to TIP.

## **1. Background**
Due to the influence of the AC-powered grid, the luminance of indoor lighting devices will be changed sinusoidally. This phenomenon is invisible, while it is inevitably
recorded by cameras with CMOS sensors at some shutter speeds. It leads to banding artifacts and decreases the quality of the captured photos, making the visual
perception unpleasant and even impairing the performance of downstream tasks.

<img src="/Figs/reason.png" width="48%" alt=""/>    <img src="/Figs/flicker.png" width="50.97%" alt=""/>

## **2. Introduction**
DeflickerCycleGAN is a framework based on [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf) for flickering image detection and single-image de-flickering. Compared to the conventional CycleGAN, DeflickerCycleGAN employs two novel and effective loss functions, which are proposed according to the physical characteristics of flickers, i.e., the flicker loss and the gradient loss. They can significantly improve the deflickering performance. To be specific, the combination of flicker loss and gradient loss effectively alleviates the phenomenon of incomplete elimination and color shift in the original CycleGAN. In addition, DeflickerCycleGAN may also have reference value for the removal of the flickers in video and the moiré patterns.

![Overviwes of DeflickerCycleGAN](/Figs/overview.png "Fig 1: Overview of DeflickerCycleGAN")

## **3. Proposed flicker loss and gradient loss**

gradient loss is formulated as: 
![](https://latex.codecogs.com/svg.image?\bg{white}\begin{equation}&space;\vspace{2mm}&space;\mathcal{L}_{{\rm&space;grad}}(R)&space;=\mathbb{E}_{y\sim&space;p_{data}(y)}[||\nabla_n&space;y&space;-&space;\nabla_n&space;R(y)||_2]&space;&plus;&space;||\nabla_m&space;R(y)||_2],\end{equation})

$\mathcal{L}_{\rm flk}$ is proposed for keeping the average of the image column during the deflickering phase. For the mapping $R:Y\mapsto X$, $\mathcal{L}_{\rm flk}$ is defined as:
\begin{align}\label{Equ 11}
	\mathcal{L}_{{\rm flk}}(R)= \mathbb{E}_{\,y\sim p_{{\rm data}(y)}}\;\ \,[\,|\!|\,\mathbb{E}_{\,M} [\, y-R(y)\,]\,|\!|_1\,]\nonumber,
	M=\{m_0, ..., m_i\}^{\rm T}.
\end{align}
