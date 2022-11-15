# **DeflickerCycleGAN v1.0**
An offcial implement of "DeflickerCycleGAN : learning to detect and remove flickers in a single image" with [tensorflow](https://www.tensorflow.org/).
It is an interesting and practical framework for eleminate the flickers in images. The details can be found in the paper, which is submitted to TIP.

## **Background**
Due to the influence of the AC-powered grid, the luminance of indoor lighting devices will be changed sinusoidally. This phenomenon is invisible, while it is inevitably
recorded by cameras with CMOS sensors at some shutter speeds. It leads to banding artifacts and decreases the quality of the captured photos, making the visual
perception unpleasant and even impairing the performance of downstream tasks.

<div align=center><img src="/Figs/flicker.png" width="40.2%" alt=""/>  <img src="/Figs/reason.png" width="41.97%" alt=""/></div>

## **Introduction**
DeflickerCycleGAN is a framework based on [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf) for flickering image detection and single-image de-flickering. Compared to the conventional CycleGAN, DeflickerCycleGAN employs two novel and effective loss functions, which are proposed according to the physical characteristics of flickers, i.e., the **flicker loss** and the **gradient loss**. They can significantly improve the deflickering performance. To be specific, the combination of flicker loss and gradient loss effectively alleviates the phenomenon of incomplete elimination and color shift in the original CycleGAN. In addition, DeflickerCycleGAN may also have reference value for the removal of the flickers in video and the moiré patterns.

![Overviwes of DeflickerCycleGAN](/Figs/framework.png)

![Overviwes of DeflickerCycleGAN](/Figs/model.png)

## De-flickering results

![Overviwes of DeflickerCycleGAN](/Figs/syn.png)

![Overviwes of DeflickerCycleGAN](/Figs/photo.png)

## Train your model
&emsp;### Prepare the data pipeline

&ensp;you can utilize **Dataset.dataset.DataLoader()** to obtain the train and eval pipeline
&ensp;training_path=['flickering images path'. 'flicker-free_images path']
&ensp;testing_path=['flickering images path'. 'flicker-free_images path']
&ensp;```
      train_set, eval_set = Dataset.dataset.DataLoader(training_path, testing_path)
      ```
&emsp;### visualize the training 
## Evaluate the performance
 ...
### Dependencies
Note: That is the setting based on my device, you can modify the torch and torchaudio version based on your device.

Start from building the environment
```
conda create -n DFcyclegan python=3.9 anaconda
conda activate DFcyclegan
pip install -r requirements.txt
```

Start from the existing environment
```
pip install -r requirements.txt
```

### Notes

If you meet any problems about this repository, **Please contact with me by E-mail <21013082029@stu.hqu.edu.cn> or <2667392087@qq.com> and you can also ask me from the 'issue' part in Github.** 

If you improve the result based on this repository by some methods, please let me know. Thanks!
