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


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.cat = layers.Concatenate()
        self.last_conv = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')
        
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
        #down_sample      
        x     = self.conv_ins_relu( 64, 4, 2)(inputs) #(128*128*64)
        skip1 = x     
        x     = self.conv_ins_relu(128, 4, 2)(x) #(64*64*128)
        skip2 = x   
        x     = self.conv_ins_relu(256, 4, 2)(x) #(32*32*256)
        skip3 = x                  
        x     = self.conv_ins_relu(512, 4, 2)(x) #(16*16*512)
        skip4 = x       
        x     = self.conv_ins_relu(512, 4, 2)(x) #(8*8*512)
        skip5 = x      
        x     = self.conv_ins_relu(512, 4, 2, apply_drop=True)(x) #(4*4*512) 
        skip6 = x 
        x     = self.conv_ins_relu(512, 4, 2, apply_drop=True)(x) #(2*2*512)
  
        #up_sample         
        x = self.convT_ins_relu(512, 4, 2, apply_drop=True)(x) #(4*4*1024)
        x = self.cat([x, skip6])
        
        x = self.convT_ins_relu(512, 4, 2, apply_drop=True)(x) #(8*8*1024)
        x = self.cat([x, skip5])

        x = self.convT_ins_relu(512, 4, 2)(x) 
        x = self.cat([x, skip4]) #(16*16*1024)

        x = self.convT_ins_relu(256, 4, 2)(x)
        x = self.cat([x, skip3]) #32*32*512
        
        x = self.convT_ins_relu(128, 4, 2)(x) 
        x = self.cat([x, skip2]) #64*64*256

        x = self.convT_ins_relu( 64, 4, 2)(x)  
        x = self.cat([x, skip1]) #128*128*128

        x = self.last_conv(x) #256*256*3(x)      
        return x
    
     
class Discriminator(Model): 
    def __init__(self):
        super(Discriminator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.last_conv = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=self.initializer)
        self.pad = ReflectionPadding2D()
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
        x = self.conv_ins_relu( 64, 4, 2)(inputs)
        x = self.conv_ins_relu(128, 4, 2)(x)
        x = self.conv_ins_relu(256, 4, 2)(x) #64*64*128
        x = self.pad(x)
        x = self.conv_ins_relu(512, 3, 1)(x)
        x = self.pad(x)
        x = self.conv_ins_relu(512, 3, 1)(x) 
        x = self.last_conv(x)  
        return x
