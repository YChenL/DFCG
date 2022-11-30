import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


def Generator():
    #down_sample
    initializer = tf.random_normal_initializer(0., 0.02)
    skips = []  
        
    inputs = layers.Input(shape=(256, 256, 3))
        
    x = layers.Conv2D(64,5,1,padding='same',activation='relu',
                      kernel_initializer=None)(inputs) #(256*256*64)
    skips.append(x)
    x = layers.Conv2D(128,3,2,padding='same',activation='relu',
                      kernel_initializer=None)(x) #(128*128*128)  
    x = layers.Conv2D(128,3,1,padding='same',activation='relu',
                      kernel_initializer=None)(x) #(128*128*128) 
    skips.append(x)
    x = layers.Conv2D(256,3,2,padding='same',activation='relu',
                      kernel_initializer=None)(x) #(64*64*256) 
        
    x = layers.Conv2D(256,3,1,padding='same',activation='relu',
                      kernel_initializer=None)(x)
    x = layers.Conv2D(256,3,1,padding='same',activation='relu',
                      kernel_initializer=None)(x)
    x = layers.Conv2D(256,3,1,padding='same',activation='relu',
                      dilation_rate=2,kernel_initializer=None)(x)
    x = layers.Conv2D(256,3,1,padding='same',activation='relu',
                      dilation_rate=4,kernel_initializer=None)(x)
    x = layers.Conv2D(256,3,1,padding='same',activation='relu',
                      dilation_rate=8,kernel_initializer=None)(x)
    x = layers.Conv2D(256,3,1,padding='same',activation='relu',
                      dilation_rate=16,kernel_initializer=None)(x)
    x = layers.Conv2D(256,3,1,padding='same',activation='relu',
                      kernel_initializer=None)(x)
    x = layers.Conv2D(256,3,1,padding='same',activation='relu',
                      kernel_initializer=None)(x) #(64*64*256) 
        
    #up_sample
                  
    x = layers.Conv2DTranspose(128,4,2,padding='same',
                               kernel_initializer=None)(x) #(128*128*128) 
    x = layers.Concatenate()([x,skips[-1]]) #(128*128*256)
    x = layers.AveragePooling2D(2,1,padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128,3,1,padding='same',activation='relu',
                      kernel_initializer=None)(x) #(128*128*128) 
        
        
    x = layers.Conv2DTranspose(64,4,2,padding='same',
                               kernel_initializer=None)(x) #(256*256*64) 
    x = layers.Concatenate()([x,skips[0]]) #(256*256*128) 
    x = layers.AveragePooling2D(2,1,padding='same')(x)
    x = layers.Activation('relu')(x) 
    x = layers.Conv2D(32,3,1,padding='same',activation='relu',
                      kernel_initializer=None)(x) #(256*256*32) 
        

    x = layers.Conv2D(3,3,1,padding='same',activation='tanh',
                      kernel_initializer=None)(x) #256*256*3(x)      
    return keras.Model(inputs=inputs, outputs=x)
    
    
def Multi_scale_Disc():   
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    #scale 1
    conv1 = layers.Conv2D(64,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(inputs) #128*128*64
    conv2 = layers.Conv2D(128,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv1) #128*128*64
    conv3 = layers.Conv2D(256,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv2) #128*128*64
    conv4 = layers.Conv2D(512,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv3) #128*128*64
    conv5 = layers.Conv2D(1,1,1,padding='same',
                          kernel_initializer=None)(conv4) #128*128*64 activation='sigmoid',
    #scale 2   
    pool1 = layers.AveragePooling2D(pool_size=3,strides=2,padding='same')(inputs) #128*128*3
    # pool1 = layers.Conv2D(3, 2, 2, padding='same')(inputs) #128*128*3
    conv6 = layers.Conv2D(64,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(pool1) #128*128*64
    conv7 = layers.Conv2D(128,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv6) #128*128*64
    conv8 = layers.Conv2D(256,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv7) #128*128*64
    conv9 = layers.Conv2D(512,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv8) #128*128*64
    conv10 = layers.Conv2D(1,1,1,padding='same',
                          kernel_initializer=None)(conv9) #128*128*64 activation='sigmoid',
    #scale 3
    pool2 = layers.AveragePooling2D(pool_size=3,strides=2,padding='same')(inputs) #64*64*3
    pool3 = layers.AveragePooling2D(pool_size=3,strides=2,padding='same')(pool2) #64*64*3
    # pool2 = layers.Conv2D(3, 4, 4, padding='same')(inputs) #64*64*3
    conv11 = layers.Conv2D(64,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(pool3) #128*128*64
    conv12 = layers.Conv2D(128,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv11) #128*128*64
    conv13 = layers.Conv2D(256,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv12) #128*128*64
    conv14 = layers.Conv2D(512,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv13) #128*128*64
    conv15 = layers.Conv2D(1,1,1,padding='same',
                          kernel_initializer=None)(conv14) #128*128*64 activation='sigmoid
    return tf.keras.Model(inputs=inputs, outputs=[conv5, conv10, conv15])



def down_sample(filters, size, apply_drop=False, Norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    downsample = tf.keras.Sequential()
    downsample.add(tf.keras.layers.Conv2D(filters, size, strides=2, 
                                          padding='same', kernel_initializer=initializer))
    if Norm:
        downsample.add(tfa.layers.InstanceNormalization())
    if apply_drop:
        downsample.add(tf.keras.layers.Dropout(0.2))
    downsample.add(tf.keras.layers.LeakyReLU())       
    return downsample


def up_sample(filters, size, apply_drop=False, Norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    upsample = tf.keras.Sequential()
    upsample.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, 
                                                 padding='same', kernel_initializer=initializer))
    if Norm:
        upsample.add(tfa.layers.InstanceNormalization())
    if apply_drop:
        upsample.add(tf.keras.layers.Dropout(0.5))
    upsample.add(tf.keras.layers.ReLU()) #上采样建议使用relu进行激活      
    return upsample

 
def Disc():   
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    
    #scale 3
    pool2 = layers.AveragePooling2D(pool_size=3,strides=2,padding='same')(inputs) #64*64*3
    pool3 = layers.AveragePooling2D(pool_size=3,strides=2,padding='same')(pool2) #64*64*3
    # pool2 = layers.Conv2D(3, 4, 4, padding='same')(inputs) #64*64*3
    conv11 = layers.Conv2D(64,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(pool3) #128*128*64
    conv12 = layers.Conv2D(128,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv11) #128*128*64
    conv13 = layers.Conv2D(256,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv12) #128*128*64
    conv14 = layers.Conv2D(512,4,2,padding='same',activation='relu',
                          kernel_initializer=None)(conv13) #128*128*64
    conv15 = layers.Conv2D(1,1,1,padding='same',
                          kernel_initializer=None)(conv14) #128*128*64 activation='sigmoid
    return tf.keras.Model(inputs=inputs, outputs=conv15)#, pool3])  ',