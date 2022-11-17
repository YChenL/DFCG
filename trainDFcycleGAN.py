import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse, glob, os, time, warnings
from Deflkcyclegan import DFcycgan
from Dataset.dataset import DataLoader


parser = argparse.ArgumentParser(description = "DFcyclegan_trainer")
## Training Settings
parser.add_argument('--max_epoch',   type=int,    default=100,   help='Maximum number of epochs')
parser.add_argument('--batch_size',  type=int,    default=1,     help='Batch size')
parser.add_argument('--buffer_size', type=int,    default=1000,  help='Number of shuffle data')
parser.add_argument('--test_step',   type=int,    default=1,     help='Test and every [test_step] epochs')
parser.add_argument('--save_step',   type=int,    default=5,     help='Save every [save_step] epochs')
parser.add_argument('--init_lr',     type=float,  default=1e-04, help='Learning rate')
parser.add_argument("--lr_decay",    type=float,  default=0.97,  help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_path',  type=list,   default=['F:/datasets/demoDataset/train/flk_img', 
                                                           'F:/datasets/demoDataset/train/org_img'],     
                                     help='The path of the training list, [flickering img Path, flicker-free img Path]')
parser.add_argument('--eval_path',   type=list,   default=['F:/datasets/demoDataset/eval/flk_img', 
                                                          'F:/datasets/demoDataset/eval/org_img'],              
                                     help='The path of the training list, [flickering img Path, flicker-free img Path]')
parser.add_argument('--save_path',   type=str,    default="./weights",   help='Path to save the models')
parser.add_argument('--load_path',   type=str,    default="./weights/",  help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--Model_arch',  type=str,    default='samescale',   help='architecture of the generators and discriminators')
parser.add_argument('--mode',        type=str,    default='full',        help='Loss margin in AAM softmax')
parser.add_argument('--use_mix',     type=bool,   default=True,  
                                     help='whether to use the combination of MS-SSIM and L1 to calcudate Lcyc and Liden')


warnings.simplefilter("ignore")
args = parser.parse_args()

trainset, testset = DataLoader(**vars(args))
Model = DFcycgan(**vars(args))


def fit(trainset, testset, load_path, save_path, test_step, save_step, max_epoch, **kwargs):
    if load_path != 0:
        Model.load_params(load_path)
        
    i = 0 
    for epoch in range(max_epoch):
        start = time.time()
        for flk_img, org_img in trainset:   
            Model.train_model(flk_img, org_img)          
            if i % 64 == 0:
                print ('.', end='')              
            i+=1    
            
        if epoch % test_step == 0:          
            print()
            for flk_img, org_img in testset.take(1):
                Model.generate_images(flk_img, org_img)
                
        if (epoch + 1) % save_step == 0:
            Model.save_params(save_path, epoch) 
            
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))
        
              
fit(trainset, testset, **vars(args))