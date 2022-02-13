# imports
from efficientnet_pytorch_3d import EfficientNet3D
import nibabel as nib
import torch
import monai
import numpy as np
import pandas as pd
import os
import random
from skimage import io, transform
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import cv2
import seaborn as sns
import PIL
from PIL import Image
from skimage.transform import resize
from torchmetrics import AUROC
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from scipy import ndimage, misc
import glob
import nibabel as nib
from nilearn import image
import torchio as tio
import shutil
import copy
from efficientnet_pytorch import EfficientNet


def plot_performance(loss_validation, loss_training, accuracy_validation, accuracy_training):
    '''
    plot the loss and the accuracy of the training/validation/ set
    '''

    plt.plot(accuracy_training)
    plt.plot(accuracy_validation)
    plt.legend(["training accuracy", "val accuracy"])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.show()
            
    plt.plot(loss_training)
    plt.plot(loss_validation)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(["training loss", "val loss"])
    plt.show()

def find_largest_tumor(mask_nifti, h = 240, w = 240, d = 155):
    '''
    this function finds the slic with the largest tumor size based on the Feret diameter. I does this for each view (Sagittal, Coronal and Axial)
    I saved the results in a lookup table in the file "slice_lookup.csv" so that I don't have to call this function each time I do the training
    I defined 2 Dataset classes below: one that uses the lookup table and another one that calls this function evrytime just incase I needed to modify 
    '''
    max_diameter_d = 0
    max_slice_d = 0
        
    max_diameter_w = 0
    max_slice_w = 0
    
    max_diameter_h = 0
    max_slice_h = 0
    
    for i in range(d):
        x = regionprops(mask_nifti[:,:,i].astype(np.int))
        try:
            diameter = x[0].feret_diameter_max
        except:
            continue
                
        if diameter > max_diameter_d:
            val, _ = np.unique(mask_nifti[:,:,i], return_counts=True)
            if len(val) == 4:
                max_diameter_d = diameter
                max_slice_d = i
                
    for i in range(w):
        x = regionprops(mask_nifti[:,i,:].astype(np.int))
        try:
            diameter = x[0].feret_diameter_max
        except:
            continue
                
        if diameter > max_diameter_w:
            val, _ = np.unique(mask_nifti[:,i,:], return_counts=True)
            if len(val) == 4:
                max_diameter_w = diameter
                max_slice_w = i
                
    for i in range(h):
        x = regionprops(mask_nifti[i,:,:].astype(np.int))
        try:
            diameter = x[0].feret_diameter_max
        except:
            continue
                
        if diameter > max_diameter_h:
            val, _ = np.unique(mask_nifti[i,:,:], return_counts=True)
            if len(val) == 4:
                max_diameter_h = diameter
                max_slice_h = i
                
                
    return max_slice_d, max_slice_w, max_slice_h

def show_images(image, label):
    """Show image with label"""    
    fig = plt.figure()
    fig.text(.5, .001, label, ha='center')
    print(image.shape)
    plt.imshow(image.permute(1,2,0))

class Normalize(object):
    '''
    this class is used as a transformation for the inout data
    it moves the pixle values to range [0, 1]
    '''
    def __call__(self, image):
        
        return image/image.max() 

class Gaussian_Noise(object):
    '''
    this class is used as a transformation for the inout data
    it adds guassian noise to the images
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        if random.random() <= 0.6:
            gaussian = np.random.normal(self.mean, self.std, (image.shape[0],image.shape[1])) 
            return image + gaussian
        return image

class Pad(object):
    '''
    this class is used as a transformation for the inout data
    it pads the images with black pixels to match the desired size
    '''
    def __init__(self, w, h):
        self.w = w
        self.h = h
    def __call__(self, image):
        h2,w2= image.shape
        
        return np.pad(image, [(self.h-h2,0),(0,self.w-w2)], mode='constant',  constant_values=(0))

class zscore(object):
    def __call__(self, image):
        mean = np.mean(image)
        std = np.std(image)
        zscore= (image -mean) / std
       # image = stats.zscore(image, nan_policy='omit')
        return zscore