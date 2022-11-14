import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.constraints import max_norm


def down_sample(filters, size, apply_drop=False, 
                regularizer=None, constraint=None):
    
    initializer = tf.random_normal_initializer(0., 0.02)   
    downsample = tf.keras.Sequential()
    downsample.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', 
                                          kernel_initializer=initializer, kernel_regularizer=regularizer,
                                          kernel_constraint=constraint
                                         ))
    downsample.add(tfa.layers.InstanceNormalization())
    if apply_drop:
        downsample.add(tf.keras.layers.Dropout(0.2))
    downsample.add(tf.keras.layers.LeakyReLU())       
    return downsample

def up_sample(filters, size, apply_drop=False,
              regularizer=None, constraint=None):
    
    initializer = tf.random_normal_initializer(0., 0.02)   
    upsample = tf.keras.Sequential()
    upsample.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', 
                                                 kernel_initializer=initializer, kernel_regularizer=regularizer,
                                                 kernel_constraint=constraint
                                                ))
    upsample.add(tfa.layers.InstanceNormalization())
    if apply_drop:
        upsample.add(tf.keras.layers.Dropout(0.5))
    upsample.add(tf.keras.layers.ReLU())    
    return upsample


def SE_block(inputs):
    x = layers.GlobalAveragePooling2D(data_format='channels_last', keepdims=False)(inputs)
    #make chennel into 1/16
    x = layers.Dense(int((1/16)*inputs.shape[-1]), activation='relu')(x)
    x = layers.Dense(int(inputs.shape[-1]))(x)    
    x = keras.activations.sigmoid(x)
    #tensor cannot do this operation
    #need to set a comtemporary variable
    x = tf.keras.layers.Reshape((1, 1, inputs.shape[-1]))(x)
    scale = inputs*x  
    return scale


def Generator():
        #down_sample
        skips = []  
        
        inputs = layers.Input(shape=(256, 256, 3))
        
        x = down_sample(64,4)(inputs) #(128*128*64)
        skips.append(x) 
        
        x = down_sample(128,4)(x) #(64*64*128)
        skips.append(x) 
        
        
        x = down_sample(256,4)(x) #(32*32*256)
        skips.append(x)         
              
        x = down_sample(512,4)(x) #(16*16*512)
        skips.append(x) 
        
        x = down_sample(512,4)(x) #(8*8*512)
        skips.append(x) 
        
        x = down_sample(512,4,apply_drop=True)(x) #(4*4*512) 
        skips.append(x) 
        
        x = down_sample(512, 4,apply_drop=True)(x) #(2*2*512)
        skips.append(x) 
        
       
        #up_sample         
        x = up_sample(512, 4, apply_drop=True)(x) #(4*4*1024)
        x = layers.Concatenate()([x,skips[5]])
        
        x = up_sample(512, 4, apply_drop=True)(x) #(8*8*1024)
        x = layers.Concatenate()([x,skips[4]])

        x =up_sample(512, 4)(x) #
        x = layers.Concatenate()([x,skips[3]]) #(16*16*1024)

        x = up_sample(256, 4)(x)
        x = layers.Concatenate()([x,skips[2]])#32*32*512
        
        x = up_sample(128, 4)(x) # 
        x = layers.Concatenate()([x,skips[1]])#64*64*256

        x = up_sample(64, 4)(x)  
        x = layers.Concatenate()([x,skips[0]])#128*128*128

        x = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(x) #256*256*3(x)      
        return keras.Model(inputs=inputs, outputs=x)
    

def Discriminator():   
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    
    down1 = down_sample(64, 4)(inputs)
    down2 = down_sample(128, 4)(down1) #64*64*128
    down3 = down_sample(256, 4)(down2) #32*32*256
    pad1 = tf.pad(down3, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv1 = layers.Conv2D(512, kernel_size=3, strides=(1, 1), 
                          padding='valid', kernel_initializer=initializer)(pad1)
    pad2 = tf.pad(conv1, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv2 = layers.Conv2D(512, kernel_size=3, strides=(1, 1), 
                          padding='valid', kernel_initializer=initializer)(pad2)
    
    last = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer)(conv2) #默认不进行全0填充.4*4*3
    
    return tf.keras.Model(inputs=inputs, outputs=last)  
