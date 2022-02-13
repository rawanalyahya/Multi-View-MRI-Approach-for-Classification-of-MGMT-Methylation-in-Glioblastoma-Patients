from datasets.with_look_up_table import FullBrainDataset
from torchvision import transforms
import torchvision.transforms as T
import pandas as pd
from brain_classification_preperation import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

train_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/train_df.csv")
test_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/test_df.csv")
#use a look-up table to find the slices with the largest tumor without having to loop everytime 
lookup_table = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/slice_lookup.csv")


#define datasets 
dataset_train = FullBrainDataset(datafram=train_set,
                                 lookup_df=lookup_table,
                                 MRI_type="T2w",
                                 transform=transforms.Compose([
                                               Pad(240, 240),
                                               #Normalize(),
                                               #Gaussian_Noise(0, 0.02),
                                               T.ToTensor(),
                                               #T.Normalize(mean = [0.485], std = [0.229]),                           
                                               #T.RandomVerticalFlip(), 
                                               #T.RandomPerspective(0.2,p=0.4),
#                                                T.RandomHorizontalFlip(),
#                                                T.RandomRotation((0,10)),
#                                                T.RandomAdjustSharpness(sharpness_factor=3, p = 0.5)
                        ]))


fig = plt.figure()

# for i in range(0,2):
    
#     sample = dataset_train[i]
#     print(sample['image'][0].max())
#     if sample == None:
#         print("error")
#         continue
    
#     if sample['image'][1].max() == 0:
#         print("image is empty")
#         show_images(sample['image'][1], sample['label'])
#         plt.show()
#         break
# #     show_images(sample['image'][0], sample['label'])
# #     show_images(sample['image'][1], sample['label'])
# #    show_images(sample['image'][2], sample['label'])
    
#     plt.hist(x=sample['image'][0].numpy().ravel(), bins=200, range=[0, 2], color='crimson')
# #     plt.hist(x=sample['image'][1].numpy().ravel(), bins=200, range=[0, 1], color='crimson')
# #     plt.hist(x=sample['image'][2].numpy().ravel(), bins=200, range=[0, 1], color='crimson')

#     plt.title("Histogram Showing Pixel Intensities And Counts", color='crimson')
#     plt.ylabel("Number Of Pixels Belonging To The Pixel Intensity", color="crimson")
#     plt.xlabel("Pixel Intensity", color="crimson")
#     #plt.show()

sample1 = dataset_train[0]
sample2 = dataset_train[10]
sample3 = dataset_train[24]
print(sample1['image'][0].max())

plt.hist(x=sample1['image'][0].numpy().ravel(), bins=200, range=[1, int(sample1['image'][0].max())])
plt.hist(x=sample2['image'][0].numpy().ravel(), bins=200, range=[1, int(sample1['image'][0].max())])
plt.hist(x=sample3['image'][0].numpy().ravel(), bins=200, range=[1, int(sample1['image'][0].max())])

#     plt.hist(x=sample['image'][1].numpy().ravel(), bins=200, range=[0, 1], color='crimson')
#     plt.hist(x=sample['image'][2].numpy().ravel(), bins=200, range=[0, 1], color='crimson')

plt.title("Histogram Showing Pixel Intensities And Counts", color='crimson')
plt.ylabel("Number Of Pixels Belonging To The Pixel Intensity", color="crimson")
plt.xlabel("Pixel Intensity", color="crimson")
plt.show()
