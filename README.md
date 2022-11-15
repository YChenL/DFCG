# **DeflickerCycleGAN v1.0**
An offcial implement of "DeflickerCycleGAN : learning to detect an remove flickers in a single image" with [tensorflow](https://www.tensorflow.org/).
It is an interesting and practical framework for eleminate the flickers in images. The details can be found in the paper, which is submitted to TIP.

## **1. Background**
Due to the influence of the AC-powered grid, the luminance of indoor lighting devices will be changed sinusoidally. This phenomenon is invisible, while it is inevitably
recorded by cameras with CMOS sensors at some shutter speeds. It leads to banding artifacts and decreases the quality of the captured photos, making the visual
perception unpleasant and even impairing the performance of downstream tasks.

## **2. Introduction**
DeflickerCycleGAN is a framework based on [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf) for single-image de-flickering. Compared to the original CycleGAN, DeflickerCycleGAN employs two novel and effective loss functions, which are proposed according to the physical characteristics of flickers, i.e., the flicker loss and the gradient loss. They significantly improve the deflickering performance. Specifically, they effectively alleviate the phenomenon of incomplete elimination and color shift in the original CycleGAN. In addition, DeflickerCycleGAN may also have reference value for the removal of the flickers in video and the moiré patterns.

![Overviwes of DeflickerCycleGAN](/Figs/overview.png "Fig 1: Overview of DeflickerCycleGAN")
