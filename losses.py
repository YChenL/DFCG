import tensorflow as tf
import numpy as np


def _tf_fspecial_gauss(size, sigma, channels=1):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
 
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)
 
    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
 
    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)
 
    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
 
    window = g / tf.reduce_sum(g)
    return tf.tile(window, (1,1,channels,channels))
 
    
def tf_gauss_conv(img, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img.get_shape().as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0
    window = _tf_fspecial_gauss(size, sigma, ch) # window shape [size, size]
    padded_img = tf.pad(img, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    return tf.nn.conv2d(padded_img, window, strides=[1,1,1,1], padding='VALID')
 

def tf_gauss_weighted_l1(img1, img2, mean_metric=True, filter_size=11, filter_sigma=1.5):
    diff = tf.abs(img1 - img2)
    L1 = tf_gauss_conv(diff, filter_size=filter_size, filter_sigma=filter_sigma)
    if mean_metric:
        return tf.reduce_mean(L1)
    else:
        return L1

    
def tf_ssim(img1, img2, cs_map=False, mean_metric=True, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img1.get_shape().as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0
    window = _tf_fspecial_gauss(size, sigma, ch) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    padded_img1 = tf.pad(img1, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")       
    padded_img2 = tf.pad(img2, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")      
    mu1 = tf.nn.conv2d(padded_img1, window, strides=[1,1,1,1], padding='VALID') 
    mu2 = tf.nn.conv2d(padded_img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1   
    mu2_sq = mu2*mu2    
    mu1_mu2 = mu1*mu2   
    paddedimg11 = padded_img1*padded_img1
    paddedimg22 = padded_img2*padded_img2
    paddedimg12 = padded_img1*padded_img2
    sigma1_sq = tf.nn.conv2d(paddedimg11, window, strides=[1,1,1,1],padding='VALID') - mu1_sq  
    sigma2_sq = tf.nn.conv2d(paddedimg22, window, strides=[1,1,1,1],padding='VALID') - mu2_sq   
    sigma12 = tf.nn.conv2d(paddedimg12, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2   
    ssim_value = tf.clip_by_value(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)), 0, 1)
    if cs_map:         
        cs_map_value = tf.clip_by_value((2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2), 0, 1)   
        value = (ssim_value, cs_map_value)
    else:
        value = ssim_value
    if mean_metric:            
        value = tf.reduce_mean(value)
    return value
  
    
def tf_ms_ssim(img1, img2, weights=None, mean_metric=False):
    if weights is None:
        weights = [1, 1, 1, 1, 1] # [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], [1, 1, 1, 1, 1] 
    level = len(weights)
    sigmas = [0.5]
    for i in range(level-1):
        sigmas.append(sigmas[-1]*2)
    weight = tf.constant(weights, dtype=tf.float32) 
    mssim = []
    mcs = []
    for l, sigma in enumerate(sigmas):
        filter_size = int(max(sigma*4+1, 11))
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size, filter_sigma=sigma)
        mssim.append(ssim_map)
        mcs.append(cs_map)
    # list to tensor of dim D+1
    value = mssim[level-1]**weight[level-1]
    for l in range(level):
        value = value * (mcs[l]**weight[l])
    if mean_metric:
        return tf.reduce_mean(value)
    else:
        return value
 

# calculate L-mix; L-mix = (1-ms_ssim_map) * alpha + l1_map * (1-alpha)
def tf_ms_ssim_l1_loss(img1, img2, mean_metric=True, alpha=0.84):
    ms_ssim_map = tf_ms_ssim(img1, img2, mean_metric=False)
    l1_map = tf_gauss_weighted_l1(img1, img2, mean_metric=False, filter_size=33, filter_sigma=8.0)
    loss_map = (1-ms_ssim_map) * alpha + l1_map * (1-alpha)
    if mean_metric:
        return tf.reduce_mean(loss_map)
    else:
        return loss_map


def flicker_loss(img1, img2): 
    avg_row_img1 = tf.reduce_mean(img1, axis=1, keepdims=True)
    avg_row_img2 = tf.reduce_mean(img2, axis=1, keepdims=True)
    k = tf.reduce_mean(tf.pow(tf.pow(avg_row_img2-avg_row_img1, 2), 0.5))    
    return k



def flicker_loss_complex(img1, img2): 
    mean_rgb1 = tf.reduce_mean(img1, axis=1, keepdims=True)
    mr1, mg1, mb1 = tf.split(mean_rgb1, 3, axis=-1)   
    mean_rgb2 = tf.reduce_mean(img2, axis=1, keepdims=True)
    mr2, mg2, mb2 = tf.split(mean_rgb2, 3, axis=-1)   
    k = tf.pow(tf.pow(mr2-mr1, 2) + tf.pow(mg2-mg1, 2)+tf.pow(mb2-mb1, 2), 0.5)      
    return k


def gradient_loss(gen_img, real_img):
    h_x = gen_img.shape[2] 
    v_x = gen_img.shape[1]
    count_v = v_x * (h_x - 1)
    h_tv_gen  = gen_img[:,:,1:,:] -gen_img[:,:,:h_x-1,:]
    h_tv_real = real_img[:,:,1:,:]-real_img[:,:,:h_x-1,:]
    h_tv = tf.reduce_mean(tf.pow((h_tv_gen - h_tv_real),2))    
    v_tv = tf.math.reduce_std(gen_img[:,1:,:,:]-gen_img[:,:v_x-1,:,:])
    return h_tv + (v_tv/count_v)


def get_mid(real, fake):
    t=tf.random.uniform(shape=[], minval=0., maxval=1.)
    exp_t=tf.fill(real.shape,t)
    mid = exp_t*real+(tf.ones_like(real)-exp_t)*fake
    return mid


def disc_loss(D_real, D_fake):
    n_scale = len(D_real)
    loss = []
    real_loss = 0
    fake_loss = 0
    for i in range(n_scale) :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_real[i]), logits=D_real[i]))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_fake[i]), logits=D_fake[i]))  
        loss.append(real_loss + fake_loss)   
    return 0.5*(sum(loss)/n_scale)



def disc_loss_gp(real_A, fake_A, GP, lambda1=0.2): #, 
    n_scale = len(real_A)
    gp = tf.math.reduce_mean(tf.math.pow((tf.norm(GP, ord=2)-1), 2))
    real_loss = 0 
    fake_loss = 0
    loss = []
    for i in range(n_scale) :
        real_loss = -tf.math.reduce_mean(real_A[i])
        fake_loss = tf.math.reduce_mean(fake_A[i])
        loss.append(real_loss + fake_loss)     
    return 0.5*(sum(loss)/n_scale)+lambda1*gp


def gen_loss(fake_A):
    n_scale = len(fake_A)
    loss = []
    fakeA_loss = 0
    for i in range(n_scale):
        fakeA_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_A[i]), logits=fake_A[i]))
        loss.append(fakeA_loss)

    return sum(loss)


def gen_loss_gp(fake_A):
    n_scale = len(fake_A)
    loss = []
    fake_loss = 0
    for i in range(n_scale):
        fake_loss = -tf.reduce_mean(fake_A[i])
        loss.append(fake_loss)
    return sum(loss)/n_scale


def cycle_loss(real_image, cycled_image):
    return tf.abs(real_image - cycled_image)


def cycle_loss_mix(real_image, cycled_image):
    return tf_ms_ssim_l1_loss(real_image, cycled_image, alpha=0.84)


def identity_loss(real_image, same_image):
    return tf.abs(real_image - same_image)


def identity_loss_mix(real_image, same_image):
    return tf_ms_ssim_l1_loss(real_image, same_image, alpha=0.84)
