import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import Model, layers, Sequential



class ReflectionPadding2D(Model):
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = padding
        self.input_spec = [layers.InputSpec(ndim=4)]
    
    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])
    
    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

    
    
class res_blk(Model):
    def __init__(self, filters=512, size=3, **kwargs):
        super(res_blk, self).__init__(**kwargs)
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.pad = ReflectionPadding2D((2, 2))
        self.conv_blk = Sequential([self.pad,
                                    layers.Conv2D(filters, size, strides=1, 
                                                  kernel_initializer=self.initializer),
                                    tfa.layers.InstanceNormalization(),
                                    layers.LeakyReLU(0.02),
                                    layers.Conv2D(filters, size, strides=1, 
                                                  kernel_initializer=self.initializer),
                                    tfa.layers.InstanceNormalization(),
                                    layers.LeakyReLU(0.02)])
              
    def call(self, inputs):
        skip = inputs
        x = self.conv_blk(inputs)
        x = x + skip
        return x
 

class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.avg = layers.AveragePooling2D(2, 1, padding='same')
        self.pad = ReflectionPadding2D((3, 3))
        self.resblk = res_blk()
        
    def stem(self, filters=64, size=7, act='leakrelu'):
        stem = Sequential()
        stem.add(self.pad)
        stem.add(layers.Conv2D(filters, size, strides=1, kernel_initializer=self.initializer)) 
        stem.add(tfa.layers.InstanceNormalization()) 
        if act == 'relu':
            stem.add(layers.ReLU())
        else:    
            stem.add(layers.LeakyReLU(0.02))       
        return stem
    
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
        x = self.stem(128)(inputs) 
        x = self.conv_ins_relu(256, 3, 2, act='lrelu')(x)
        x = self.conv_ins_relu(512, 3, 2, act='lrelu')(x)
        x = self.resblk(x)
        x = self.resblk(x)
        x = self.resblk(x)
        x = self.resblk(x)
        x = self.resblk(x)
        x = self.resblk(x)
        x = self.resblk(x)
        x = self.resblk(x)
        x = self.convT_ins_relu(256, 3, 2, act='lrelu')(x)
        x = self.convT_ins_relu(128, 3, 2, act='lrelu')(x)
        x = self.stem(3)(x)
        return x
    
class Discriminator(Model):
    def __init__(self):
        super(Disc, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.resblk = res_blk()
        self.last_conv = layers.Conv2D(1, 4, 1, padding='same')
            
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
        x = self.conv_ins_relu( 64, 4, 2, Norm=False)(inputs)
        x = self.conv_ins_relu(128, 4, 2)(x)
        x = self.conv_ins_relu(256, 4, 2)(x)
        x = self.conv_ins_relu(512, 4, 2)(x)
        x = self.resblk(x)
        x = self.last_conv(x)
        return x