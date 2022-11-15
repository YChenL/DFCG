# **DeflickerCycleGAN v1.0**
An offcial implement of "DeflickerCycleGAN : learning to detect an remove flickers in a single image" with [tensorflow](https://www.tensorflow.org/).
It is an interesting and practical framework for eleminate the flickers in images. The details can be found in the paper, which is submitted to TIP.

## **1. Introduction**
Due to the influence of the AC-powered grid, the luminance of indoor lighting devices will be changed sinusoidally. This phenomenon is invisible, while it is inevitably recorded by cameras with CMOS sensors at some shutter durations. It leads to banding artifacts and decreases the quality of the captured photos, making the visual perception unpleasant and even impairing the performance of downstream tasks.

![Overviwes of DeflickerCycleGAN](/Figs/overview.png "Fig 1: Overview of DeflickerCycleGAN")
