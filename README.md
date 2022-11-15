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

## Prepare the data pipeline
you can utilize **Dataset.dataset.DataLoader( )** to obtain the train and eval pipeline.
```
training_path=['flickering images path'. 'flicker-free_images path']
testing_path=['flickering images path'. 'flicker-free_images path']
train_set, eval_set = Dataset.dataset.DataLoader(training_path, testing_path)
```
and you can also utilize **Dataset.synthesize_data.synthsize( )** to synthetic the flickering images accordings to the characteristics of flicker.
```
syn_img = synthsize(file_path, model=0)
# model 0 —> lighting conditions: Flourescent Light; model 1 —> lighting conditions: LED
```
## Train your model
### training
you can utilize **Deflkcyclegan.DFcycgan.train_model( )** to begin the training.
```
for epochs in range(args.EPOCHS):
      for flk_img, flk_free_img in train_set:
            Deflkcyclegan.DFcycgan.train_model(flk_img, flk_free_img)      
```
### visualization
you can utilize **Deflkcyclegan.DFcycgan.generate_images( )** to visualize the training processing.
```
for flk_img, flk_free_img in eval_set.take(n)
     Deflkcyclegan.DFcycgan.generate_images(flk_img, flk_free_img)     
```
### save & load checkpoints
```
Deflkcyclegan.DFcycgan.save_params(save_path) # save the checkpoints
Deflkcyclegan.DFcycgan.load_params(load_path) # load the checkpoints
```
## Evaluate the performance
### Remove & Generate the flicker in a single image
```
result = Deflkcyclegan.DFcycgan._call(img, model='rem')  
# model=='rem': remove flickers; model=='gen': generate flicker
```
### Classification
you can utilize Deflkcyclegan.DFcycgan.classify( ) & Deflkcyclegan.DFcycgan.ROC_curve( ) to achieve the classification and obtain the ROC curve (Fig.10 in the paper)
```
img, mode='full'
```
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
