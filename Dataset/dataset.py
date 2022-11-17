import tensorflow as tf
import numpy as np
import math
import glob
import os

AUTOTUNE = tf.data.AUTOTUNE

def read_image(image):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    return image


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[256, 256, 3])
    return cropped_image


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) -1
    return image


def random_jitter(image):
    image = tf.image.resize(image, [286, 286],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image


def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image):
    image = tf.image.resize(image, [256, 256],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = normalize(image)
    return image


def data_list(train_path):
    flk_list = []
    flk_list += glob.glob(os.path.join(train_path[0], '*.jpg'))
    flk_list += glob.glob(os.path.join(train_path[0], '*.jpeg'))
    flk_list += glob.glob(os.path.join(train_path[0], '*.png'))
    flk_free_list = []
    flk_free_list += glob.glob(os.path.join(train_path[1], '*.jpg'))
    flk_free_list += glob.glob(os.path.join(train_path[1], '*.jpeg'))
    flk_free_list += glob.glob(os.path.join(train_path[1], '*.png'))
    return flk_list, flk_free_list

    
    
def DataLoader(train_path, eval_path, buffer_size, batch_size, **kwargs):
    train_flk, train_flk_free = data_list(train_path)
    test_flk, test_flk_free   = data_list(eval_path)  
    
    train_flk_set      = tf.data.Dataset.from_tensor_slices(train_flk).map(read_image)
    train_flk_free_set = tf.data.Dataset.from_tensor_slices(train_flk_free).map(read_image)
    test_flk_set       = tf.data.Dataset.from_tensor_slices(test_flk).map(read_image)
    test_flk_free_set  = tf.data.Dataset.from_tensor_slices(test_flk_free).map(read_image)
    
    train_flk_set      = train_flk_set.cache().map(preprocess_image_train, 
                                                   num_parallel_calls=AUTOTUNE).shuffle(buffer_size).batch(batch_size)
    train_flk_free_set = train_flk_free_set.cache().map(preprocess_image_train, 
                                                        num_parallel_calls=AUTOTUNE).shuffle(buffer_size).batch(batch_size)
    test_flk_set       = test_flk_set.cache().map(preprocess_image_train, 
                                                  num_parallel_calls=AUTOTUNE).shuffle(buffer_size).batch(batch_size)
    test_flk_free_set  = test_flk_free_set.cache().map(preprocess_image_train, 
                                                       num_parallel_calls=AUTOTUNE).shuffle(buffer_size).batch(batch_size)
    
    return tf.data.Dataset.zip((train_flk_set, train_flk_free_set)), tf.data.Dataset.zip((test_flk_set, test_flk_free_set))
