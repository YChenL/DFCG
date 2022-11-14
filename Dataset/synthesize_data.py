import glob
import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def imread_1c(image):
    image=cv2.imread(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return image


def imread_3c(image):
    image=cv2.imread(image)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image


def make_matrix_3D():
    a = np.linspace(0,0.13,256)
    s = np.random.uniform(low=0.0, high=2.0*math.pi, size=None)
    w = np.random.uniform(low=98, high=102, size=None)
    b = 0.6*np.sin(w*a+s)
    # 1Darray —> 2Darray
    c = []
    for i in range(256):
        c.append(b)

    d = tf.convert_to_tensor(c)
    d = d.numpy()
    e = d.transpose(1, 0)

    # 2Darray —> 3Darray
    f = []
    for i in range(3):
        f.append(e)
        
    g = tf.convert_to_tensor(f)
    h = g.numpy()
    k = h.transpose(1,2,0)
    
    # only add dark parts
    # k_2 = tf.nn.relu(k)
    # k_3 = (-k_2).numpy()
    # k = abs(k)
    # k = k-0.8
        
    return k


def make_matrix_2D(p):
    '''
     p=f_enf/f_row
    '''
    m = np.linspace(0,255,256)
    phi = np.random.uniform(low=0.0, high=2.0*math.pi, size=None)
    flk_1d = 1*np.cos(4.0*math.pi*p*m+phi)
    
    # 1Darray —> 2Darray
    c = []
    for i in range(256):
        c.append(flk_1d)

    d = tf.convert_to_tensor(c)
    d = d.numpy()
    e = d.transpose(1, 0)
    e = np.expand_dims(e, axis=2)
    
    # only add dark parts   
    # e_2 = tf.nn.relu(e)
    # e_3 = (-e_2).numpy()
    # e = abs(e)
    
    return e


def read_ENF_img_3c(path, L1, L2, L3, p):
    '''
     path: str, data path to be synthesize
     L1: flickering intensity in R (1/A^c, c=R)
     L2: flickering intensity in G (1/A^c, c=G)
     L3: flickering intensity in B (1/A^c, c=B)
    '''
    image = imread_3c(path)
    image = image.astype(np.float64)
    image = image/255.
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    # make 2D flickering matrix
    e = make_matrix_2D(p)
    
    # Add flickering componets to associated channel according to the spectrum of light
    r,g,b = cv2.split(image)
    r_2 = cv2.add(e/L1,r)
    g_2 = cv2.add(e/L2,g)
    b_2 = cv2.add(e/L3,b)
    
    img = cv2.merge([r_2,g_2,b_2])
    ENF = cv2.merge([e/L1,e/L2,e/L3])
    
    image = image*255.
    img = img*255.
    ENF = (ENF*0.5+0.5)*255.
    
    cv2.imwrite("C:/Users/LYC/Desktop/synthsis/syn/n3" + str(p) + ".jpg", img)
    cv2.imwrite("E:/datasets/image/org_img/" + str(i) + ".jpg", image)
    cv2.imwrite("C:/Users/LYC/Desktop/synthsis/enf/n3" + str(p) + "ENF.jpg", ENF)

def synthsize(root, model=0):
    '''
     root: list, ele is the path of the image to be synthesized
     model: 0 —> Flourescent Light, else —> LED
    '''
    for i in range(0,len(root)):
    p = tf.truncated_normal((1,), mean=5.25e-3, stddev=7.5e-4, dtype=tf.float32, seed=None, name=None).numpy()
    # L = I/2
    L = tf.truncated_normal((1,), mean=4.5, stddev=0.5, dtype=tf.float32, seed=None, name=None).numpy()
    
    # Flourescent Light
    if model==0:
        l1 = 1*L
        l2 = 3*L
        l3 = 2*L    
        read_ENF_img_3c(root[i], l1, l2, l3, p)
        
    #  LED    
    else :
        l1 = 2*L
        L2 = 2*L
        l3 = 2*L 
        read_ENF_img_3c(root[i], l1, l2, l3, p)    