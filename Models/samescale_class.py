import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import Model, layers, Sequential


def SE_block(inputs):
    x = layers.GlobalAveragePooling2D(data_format='channels_last', keepdims=False)(inputs)
    x = layers.Dense(int((1/16)*inputs.shape[-1]), activation='relu')(x)
    x = layers.Dense(int(inputs.shape[-1]))(x)    
    x = keras.activations.sigmoid(x)
    x = tf.keras.layers.Reshape((1, 1, inputs.shape[-1]))(x)
    scale = inputs*x  
    return scale


class Generator(Model):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.cat = layers.Concatenate()
        self.avg = layers.AveragePooling2D(2, 1, padding='same')
        self.last_conv = layers.Conv2D(3, 3, 1, padding='same',activation='tanh',
                                       kernel_initializer=self.initializer)   
           
    def conv_ins_relu(self, filters, size, stride, dr=1,
                      Norm=True, apply_drop=False, act='LeakyRelu'):
        conv_block = Sequential()
        conv_block.add(layers.Conv2D(filters, size, stride, padding='same', dilation_rate=(dr, dr), 
                                     kernel_initializer=self.initializer))
        if Norm:
            conv_block.add(tfa.layers.InstanceNormalization())
        if apply_drop:
            conv_block.add(layers.Dropout(0.2))
        if act=='Relu':
            conv_block.add(layers.ReLU())    
        else:
            conv_block.add(layers.LeakyReLU(0.02))       
        return conv_block

    def convT_ins_relu(self, filters, size, stride, dr=1,
                       Norm=True, apply_drop=False, act='Relu'): 
        convT_block = Sequential()
        convT_block.add(layers.Conv2DTranspose(filters, size, stride, padding='same', dilation_rate=(dr, dr), 
                                               kernel_initializer=self.initializer))
        if Norm:
            convT_block.add(tfa.layers.InstanceNormalization())
        if apply_drop:
            convT_block.add(layers.Dropout(0.2))
        if act=='Relu':
            convT_block.add(layers.ReLU()) 
        else:
            convT_block.add(layers.LeakyReLU(0.02)) 
        return convT_block
    
    def call(self, inputs):
        # subsampling
        x = self.conv_ins_relu( 64, 5, 1)(inputs) # 256*256*64
        skip1 = x
        x = self.conv_ins_relu(128, 3, 2)(x)      # 128*128*128
        x = self.conv_ins_relu(128, 3, 1)(x)      # 128*128*128
        skip2 = x
        x = self.conv_ins_relu(256, 3, 2)(x)      # 64*64*256
        res1 = x
        x = self.conv_ins_relu(256, 3, 1)(x)
        x = self.conv_ins_relu(256, 3, 1)(x)
        x = x + res1
        res2 = x
        x = self.conv_ins_relu(256, 3, 1,  2)(x)
        x = self.conv_ins_relu(256, 3, 1,  4)(x)
        x = x + res2
        res3 = x
        x = self.conv_ins_relu(256, 3, 1,  8)(x)
        x = self.conv_ins_relu(256, 3, 1, 16)(x)
        x = x + res3
        res4 = x
        x = self.conv_ins_relu(256, 3, 1)(x)
        x = self.conv_ins_relu(256, 3, 1)(x)
        x = x + res4
        
        # upsampling
        x = self.convT_ins_relu(128, 4, 2)(x)   # 128*128*128
        x = self.cat([x,skip2])
        x = self.avg(x)
        x = self.conv_ins_relu(128, 3, 1)(x)
        
        x = self.convT_ins_relu(64, 4, 2)(x)    # 256*256*64
        x = self.cat([x,skip1])
        x = self.avg(x)
        x = self.conv_ins_relu(32, 3, 1)(x)
        
        x = self.last_conv(x)
        return x
    
        
        
class Multi_scale_Disc(Model): 
    def __init__(self, **kwargs):
        super(Multi_scale_Disc, self).__init__(**kwargs)
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.avg = layers.AveragePooling2D(3, 2, padding='same')
        self.last_conv = layers.Conv2D(1, 1, 1, padding='same',
                                       kernel_initializer=self.initializer)   
        
    def conv_ins_relu(self, filters, size, stride, dr=1,
                      Norm=True, apply_drop=False, act='LeakyRelu'):
        conv_block = Sequential()
        conv_block.add(layers.Conv2D(filters, size, stride, padding='same', dilation_rate=(dr, dr), 
                                     kernel_initializer=self.initializer))
        if Norm:
            conv_block.add(tfa.layers.InstanceNormalization())
        if apply_drop:
            conv_block.add(layers.Dropout(0.2))
        if act=='Relu':
            conv_block.add(layers.ReLU())    
        else:
            conv_block.add(layers.LeakyReLU(0.01))       
        return conv_block      
    
    def call(self, inputs):
        #scale1
        x1 = self.conv_ins_relu( 64, 4, 2)(inputs)
        x1 = self.conv_ins_relu(128, 4, 2)(x1)
        x1 = self.conv_ins_relu(256, 4, 2)(x1)
        x1 = self.conv_ins_relu(512, 4, 2)(x1)
        x1 = self.last_conv(x1) 
        #scale2
        x2 = self.avg(inputs)
        x2 = self.conv_ins_relu( 64, 4, 2)(x2)
        x2 = self.conv_ins_relu(128, 4, 2)(x2)
        x2 = self.conv_ins_relu(256, 4, 2)(x2)
        x2 = self.conv_ins_relu(512, 4, 2)(x2)
        x2 = self.last_conv(x2)
        #scale3
        x3 = self.avg(inputs) 
        x3 = self.avg(x3)
        x3 = self.conv_ins_relu( 64, 4, 2)(x3)
        x3 = self.conv_ins_relu(128, 4, 2)(x3)
        x3 = self.conv_ins_relu(256, 4, 2)(x3)
        x3 = self.conv_ins_relu(512, 4, 2)(x3)
        x3 = self.last_conv(x3)
        return [x1, x2, x3]