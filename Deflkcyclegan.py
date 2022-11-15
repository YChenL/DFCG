import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from losses import gen_loss, disc_loss, identity_loss, identity_loss_mix, \
                   cycle_loss, cycle_loss_mix, flicker_loss, gradient_loss
from Models import samescale, resnet, unet
from utils import read_img, exp_dim




class DFcycgan(Model):
    def __init__(self, Model_arch, initial_learning_rate = 1e-04, 
                 model_name = ['G', 'R', 'Dx', 'Dy', 'optms_G', 'optms_R', 'optms_Dx', 'optms_Dy']):
        super(DFcycgan, self).__init__()
        if  Model_arch == 'samescale':          
            self.G  = samescale.Generator()    
            self.R  = samescale.Generator()
            self.Dx = samescale.Multi_scale_Disc()
            self.Dy = samescale.Multi_scale_Disc()
        elif Model_arch == 'resnet':
            self.G  = resnet.Generator()
            self.R  = resnet.Generator()
            self.Dx = resnet.Discriminator()
            self.Dy = resnet.Discriminator()    
        elif Model_arch == 'unet':
            self.G  = unet.Generator()
            self.R  = unet.Generator()
            self.Dx = unet.Discriminator()
            self.Dy = unet.Discriminator()        
        else:
            raise RuntimeError("incorrectly acrhitecture!")
            
        self.g_loss = gen_loss
        self.d_loss = disc_loss
        self.iden_loss = identity_loss
        self.cyc_loss  = cycle_loss
        self.flk_loss  = flicker_loss
        self.grad_loss = gradient_loss
        self.iden_loss_mix = identity_loss_mix
        self.cyc_loss_mix  = cycle_loss_mix
        self.lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate, 
                                                                 decay_steps=100000, 
                                                                 decay_rate=0.96, 
                                                                 staircase=True)
        self.optms_G  = optimizers.Adam(self.lr_schedule, beta_1=0.5)
        self.optms_R  = optimizers.Adam(self.lr_schedule, beta_1=0.5)
        self.optms_Dx = optimizers.Adam(self.lr_schedule, beta_1=0.5)
        self.optms_Dy = optimizers.Adam(self.lr_schedule, beta_1=0.5)
        self.model_list = [self.G, self.R, self.Dx, self.Dy, self.optms_G, 
                           self.optms_R, self.optms_Dx, self.optms_Dy]
        self.model_name = model_name
        
    @tf.function
    def train_model(self, real_x, real_y, use_mix=True):
        with tf.GradientTape(persistent=True) as tape:
            '''
             real_x: flickering images, real_y : flicker-free images
             G : Y --> X
             R : X --> Y
            '''        
            fake_x = self.G(real_y, training=True)
            fake_y = self.R(real_x, training=True)
            cyc_x  = self.G(fake_y, training=True)
            cyc_y  = self.R(fake_x, training=True)
            same_x = self.G(real_x, training=True)      
            same_y = self.R(real_y, training=True)    
            
            disc_real_x = self.Dx(real_x, training=True)
            disc_fake_x = self.Dx(fake_x, training=True)     
            disc_real_y = self.Dy(real_y, training=True)
            disc_fake_y = self.Dy(fake_y, training=True)                
            # adv loss
            gen_loss_G = self.g_loss(disc_fake_x)
            gen_loss_R = self.g_loss(disc_fake_y)
            disc_x_loss = self.d_loss(disc_real_x, disc_fake_x) 
            disc_y_loss = self.d_loss(disc_real_y, disc_fake_y)     
            
            if use_mix:           
                # cycle loss
                total_cycle_loss = self.cyc_loss_mix(real_x, cyc_x) + self.cyc_loss_mix(real_y, cyc_y)      
                # identity loss 
                iden_G = self.iden_loss_mix(real_x, same_x)
                iden_R = self.iden_loss_mix(real_y, same_y)    
            else:
                # cycle loss
                total_cycle_loss = self.cyc_loss(real_x, cyc_x) + self.cyc_loss(real_y, cyc_y)      
                # identity loss 
                iden_G = self.iden_loss(real_x, same_x)
                iden_R = self.iden_loss(real_y, same_y) 
                
            # flicker loss
            flk_loss_G = flicker_loss(real_y, fake_x)
            flk_loss_R = flicker_loss(real_x, fake_y) 
            # gradient loss
            grad_loss = gradient_loss(fake_y, real_x)
            
            total_G_loss = 1*gen_loss_G + 5*iden_G + 10*total_cycle_loss + 1*flk_loss_G 
            total_R_loss = 1*gen_loss_R + 5*iden_R + 10*total_cycle_loss + 1*flk_loss_R + 100*grad_loss                          
    
        # Calculate the gradients
        G_gradients  = tape.gradient(total_G_loss, self.G.trainable_variables)
        R_gradients  = tape.gradient(total_R_loss, self.R.trainable_variables) 
        Dx_gradients = tape.gradient(disc_x_loss, self.Dx.trainable_variables)
        Dy_gradients = tape.gradient(disc_y_loss, self.Dy.trainable_variables)
    
        # optimize
        self.optms_G.apply_gradients( zip(G_gradients,   self.G.trainable_variables))
        self.optms_R.apply_gradients( zip(R_gradients,   self.R.trainable_variables)) 
        self.optms_Dx.apply_gradients(zip(Dx_gradients, self.Dx.trainable_variables))
        self.optms_Dy.apply_gradients(zip(Dy_gradients, self.Dy.trainable_variables))
    
           
    def _call(self, inputs, mode):
        if mode =='rem':
            x = self.R(inputs)
        elif mode =='gen':
            x = self.G(inputs)      
        return x
    
    
    def ROC_curve(tp, fp):
        curve = []
        for i in range(0, 1, 0.1):
            FPR = np.sum(fp>=i)/len(fp)
            TPR = np.sum(tp>=i)/len(tp)
            curve.append([FRR, TPR])
        return curve
 

    def classify(self, img_path, mode='full'):
        sig_Dx, sig_Dy, D1, D2, DF = [], [], [], [], []
        org_img = read_img(img_path)
        org_img = exp_dim(org_img, 0)
        decision_x = self.Dx(org_img) 
        decision_y = self.Dy(org_img)     
        scale = len(decision_x)      
        for i in range(scale):
            sig_Dx.append(tf.keras.activations.sigmoid(decision_x[i]))
            sig_Dy.append(tf.keras.activations.sigmoid(decision_y[i])) 
            Dx_prime = tf.math.reduce_mean(decision_x[i])
            Dy_prime = tf.math.reduce_mean(decision_y[i])
            D1.append(Dx_prime)
            D2.append(Dy_prime)     
     
        if model='full':
            DX = (tf.math.reduce_mean(sig_Dx[0])+tf.math.reduce_mean(sig_Dx[1])+tf.math.reduce_mean(sig_Dx[2]))/3
            DY = (tf.math.reduce_mean(sig_Dy[0])+tf.math.reduce_mean(sig_Dy[1])+tf.math.reduce_mean(sig_Dy[2]))/3
            DF.append(D1)
            DF.append(D2)    
        elif model='(4)':
            DX = tf.math.reduce_mean(sig_Dx[2])
            DY = tf.math.reduce_mean(sig_Dy[2])
            DF.append(D1[2])
            DF.append(D2[2])
        elif model='(8)':
            DX = tf.math.reduce_mean(sig_Dx[1])
            DY = tf.math.reduce_mean(sig_Dy[1])
            DF.append(D1[1])
            DF.append(D2[1])
        elif model='(16)':
            DX = tf.math.reduce_mean(sig_Dx[0])
            DY = tf.math.reduce_mean(sig_Dy[0])
            DF.append(D1[0])
            DF.append(D2[0])
        elif model='(4,8)':
            DX = (tf.math.reduce_mean(sig_Dx[1])+tf.math.reduce_mean(sig_Dx[2]))/2
            DY = (tf.math.reduce_mean(sig_Dy[1])+tf.math.reduce_mean(sig_Dy[2]))/2
            DF.append(D1[1:])
            DF.append(D2[1:])
        elif model='(4,16)':
            del D1[1], D2[1]     
            DX = (tf.math.reduce_mean(sig_Dx[0])+tf.math.reduce_mean(sig_Dx[2]))/2
            DY = (tf.math.reduce_mean(sig_Dy[0])+tf.math.reduce_mean(sig_Dy[2]))/2
            DF.append(D1)
            DF.append(D2)
        elif model='(8,16)':
            DX = (tf.math.reduce_mean(sig_Dx[0])+tf.math.reduce_mean(sig_Dx[1]))/2
            DY = (tf.math.reduce_mean(sig_Dy[0])+tf.math.reduce_mean(sig_Dy[1]))/2
            DF.append(D1[:-1])
            DF.append(D2[:-1])
        else:
            raise RuntimeError("incorrectly classification model!")
            
        DF = tf.convert_to_tensor(DF) 
        DF = tf.nn.softmax(DF).numpy()   
        '''
         for Dx:
            TP: DX >= threshold
            FP: DX >= threshold
            TN: DX <  threshold
            FN: DX <  threshold
                 
         for Dy:
            TP: DY  < threshold
            FP: DY  < threshold
            TN: DY >= threshold
            FN: DT >= threshold

         for Dx + Dy:
            TP: DF[:,0] >= threshold
            FP: DF[:,0] >= threshold
            TN: DF[:,0] <  threshold
            FN: DF[:,0] <  threshold    
        '''
        return DX, DY, DF 
    
        
    def save_params(self, save_path, epoch=0):
        i = 0
        for model in self.model_list:
            ckpt = tf.train.Checkpoint(model)
            ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(save_path, self.model_name[i]), max_to_keep=5)
            model_sp = ckpt_manager.save()            
            print (self.model_name[i] + ': Saving checkpoint for epoch {} at {}'.format(epoch+1, model_sp))
            i += 1
  

    def load_params(self, load_path):
        i = 0
        for model in self.model_list:
            ckpt = tf.train.Checkpoint(model)
            ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(load_path, self.model_name[i]), max_to_keep=5)
            try:
                ckpt.restore(ckpt_manager.latest_checkpoint)
                print (self.model_name[i] + ': Latest checkpoint restored!!')
            except:
                print (self.model_name[i] + ': non existing checkpoint!!')
                # raise ValueError("non existing checkpoint!!")  
            i += 1
    
            
    def generate_images(self, test_x, test_y):
        rem = self.R(test_x)
        gen = self.G(test_y)
        plt.figure(figsize=(12, 12))
        display_list = [test_x[0], rem[0], test_y[0], gen[0]]
        title = ['Input ENF', 'Predicted Rem', 'Input nonENF', 'Predicted Gen']

        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()