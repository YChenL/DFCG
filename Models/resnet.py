import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


def stem(x, filters=64, size=7):
    initializer = tf.random_normal_initializer(0., 0.02)
    # inputs = layers.Input(shape=(input_shape[0], input_shape[1], input_shape[-1]))
    pad = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
    x1 = layers.Conv2D(filters, size, strides=1, kernel_initializer=initializer)(pad)   
    x2 = tfa.layers.InstanceNormalization()(x1) 
    x3 = layers.LeakyReLU(alpha=0.2)(x2) 
    # x3 = layers.Activation('relu')(x2) 
    return x3
 
    
def down_sample(x, filters, size=3, leakyReLU=False, Norm=True):        
    initializer = tf.random_normal_initializer(0., 0.02)
    # inputs = layers.Input(shape=(input_shape[0], input_shape[1], input_shape[-1]))
    pad = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    x1 = layers.Conv2D(filters, size, strides=2, kernel_initializer=initializer)(pad)
    if Norm:
        x2 = tfa.layers.InstanceNormalization()(x1)
    else:
        x2 = x1
    
    if leakyReLU:
        x3 = layers.LeakyReLU(alpha=0.2)(x2)
    else:
        x3 = layers.Activation('relu')(x2)                        
    return x3


def up_sample(x, filters, size=3):        
    initializer = tf.random_normal_initializer(0., 0.02)
    # inputs = layers.Input(shape=(input_shape[0], input_shape[1], input_shape[-1]))
    # pad = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    x1 = layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer)(x)
    x2 = tfa.layers.InstanceNormalization()(x1)
    x3 = layers.Activation('relu')(x2)                        
    return x3

  
    
def resblk(x, filters=512, size=3):        
    initializer = tf.random_normal_initializer(0., 0.02)
    # inputs = layers.Input(shape=(input_shape[0], input_shape[1], input_shape[-1]))
    pad1 = tf.pad(x, [[0,0],[2,2],[2,2],[0,0]], "REFLECT")
    x1 = layers.Conv2D(filters, size, strides=1, kernel_initializer=initializer)(pad1)
    x2 = tfa.layers.InstanceNormalization()(x1)
    x3 = layers.LeakyReLU(alpha=0.2)(x2) 
    # x3 = layers.Activation('relu')(x2)
    x4 = layers.Conv2D(filters, size, strides=1, kernel_initializer=initializer)(x3)
    x5 = tfa.layers.InstanceNormalization()(x4)
    x6 = layers.LeakyReLU(alpha=0.2)(x5) 
    # x6 = layers.Activation('relu')(x5)
    res2 =layers.Add()([x, x6])       
    return res2


def Generator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = stem(inputs, 128)
    x = down_sample(x, 256, leakyReLU=True)
    x = down_sample(x, 512, leakyReLU=True)
    x = resblk(x)
    x = resblk(x)
    x = resblk(x)
    x = resblk(x)
    x = resblk(x)
    x = resblk(x)
    x = resblk(x)
    x = resblk(x)
    x = up_sample(x, 256)
    x = up_sample(x, 128)
    last = stem(x, 3)
    return keras.Model(inputs=inputs, outputs=last)


def Discriminator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = down_sample(inputs, 64, 4, leakyReLU=True, Norm=False)
    x = down_sample(x, 128, 4, leakyReLU=True)
    x = down_sample(x, 256, 4, leakyReLU=True)
    x = down_sample(x, 512, 4, leakyReLU=True)
    x = resblk(x)
    last = layers.Conv2D(1, 4, 1, padding='same')(x)
    return keras.Model(inputs=inputs, outputs=last)