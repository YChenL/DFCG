import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Dataset.dataset import random_jitter


def read_img(path):
    file = cv2.imread(path)
    file = cv2.cvtColor(file,cv2.COLOR_BGR2RGB)
    file = cv2.resize(file, (256, 256), interpolation=cv2.INTER_LINEAR)
    file = (file/127.5) - 1
    file = file.astype('float32')
    return file


def exp_dim(img, axis):
    img = np.expand_dims(img, axis=axis)
    return img


def dummy_loader(model_path):
    '''
    Load a stored keras model and return its weights.
    
    Input
    ----------
        The file path of the stored keras model.
    
    Output
    ----------
        Weights of the model.
        
    '''
    backbone = keras.models.load_model(model_path, compile=False)
    W = backbone.get_weights()
    return W

def image_to_array(filenames, size, channel):
    '''
    Converting RGB images to numpy arrays.
    
    Input
    ----------
        filenames: an iterable of the path of image files
        size: the output size (height == width) of image. 
              Processed through PIL.Image.NEAREST
        channel: number of image channels, e.g. channel=3 for RGB.
        
    Output
    ----------
        An array with shape = (filenum, size, size, channel)
        
    '''
    
    # number of files
    L = len(filenames)
    
    # allocation
    out = np.empty((L, size, size, channel))
    
    # loop over filenames
    if channel == 1:
        for i, name in enumerate(filenames):
            with Image.open(name) as pixio:
                pix = pixio.resize((size, size), Image.NEAREST)
                out[i, ..., 0] = np.array(pix)
    else:
        for i, name in enumerate(filenames):
            with Image.open(name) as pixio:
                pix = pixio.resize((size, size), Image.NEAREST)
                out[i, ...] = np.array(pix)[..., :channel]
    return out[:, ::-1, ...]

def shuffle_ind(L):
    '''
    Generating random shuffled indices.
    
    Input
    ----------
        L: an int that defines the largest index
        
    Output
    ----------
        a numpy array of shuffled indices with shape = (L,)
    '''
    
    ind = np.arange(L)
    np.random.shuffle(ind)
    return ind

def freeze_model(model, freeze_batch_norm=False):
    '''
    freeze a keras model
    
    Input
    ----------
        model: a keras model
        freeze_batch_norm: False for not freezing batch notmalization layers
    '''
    if freeze_batch_norm:
        for layer in model.layers:
            layer.trainable = False
    else:
        from tensorflow.keras.layers import BatchNormalization    
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    return model


def show_sample(title, sample):
    plt.subplot(121)
    plt.title(str(title))
    plt.imshow(sample[0] * 0.5 + 0.5)
    plt.axis('off')

    plt.subplot(122)
    plt.title(str(title) + ' with random jitter')
    plt.imshow(random_jitter(sample[0]) * 0.5 + 0.5)
    plt.axis('off')

