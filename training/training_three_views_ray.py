import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import sys, os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import torch.optim as optim
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

#read csv files
train_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/train_df.csv")
test_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/test_df.csv")
#use a look-up table to find the slices with the largest tumor without having to loop everytime 
lookup_table = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/slice_lookup.csv")

class FullBrainDataset(Dataset):

    def __init__(self, datafram, lookup_df, MRI_type, transform=None):
        """
        Args:
            datafram (pandas.dataframe): dataframe of dataset with labels
            lookup_df (pandas.dataframe): dataframe of look-up table
            MRI_type: which MRI type is this [FLAIR, T1wCE, T2w, T1w]
            transform: if passed apply transformation to input images
        """
       
        self.df = datafram
        self.transform = transform
        self.lookup_df = lookup_df

        self.MRI_type = MRI_type

        assert self.MRI_type in ["FLAIR", "T1wCE", "T2w", "T1w"]

    def __len__(self):
        
        return self.df.groupby("BraTS21ID").count().shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df[(self.df["BraTS21ID"] == self.df.iloc[idx*4]["BraTS21ID"]) & (self.df["mri_type"] == self.MRI_type)]["path"].values[0]

        slice = self.lookup_df[self.lookup_df["id"] == self.df.iloc[idx*4]["BraTS21ID"]]
        largest_slice_x = int(slice["x"])
        largest_slice_y = int(slice["y"])
        largest_slice_z = int(slice["z"])
        
        img_d  = nib.load(img_name).get_fdata()[:,:,largest_slice_z]     
        img_w  = nib.load(img_name).get_fdata()[:,largest_slice_y,:]
        img_h  = nib.load(img_name).get_fdata()[largest_slice_x,:,:]
        
          
        if img_d.max() == 0 or img_w.max() == 0 or img_h.max() == 0:
            return None
        
        label = self.df["MGMT_value"][int(idx)*4]
        sample = {'image':[img_h,img_w,img_d], 'label': label}
        
        if self.transform:
            for i in range(3):
                sample['image'][i]= self.transform(sample['image'][i].copy())
                
  
        return sample

class monai_densenet_three_views(nn.Module):
    def __init__(self, l1=150528//128, l2=150528//512):
        super(monai_densenet_three_views, self).__init__()
        self.model = monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features
        
        self.model2 = monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features
        
        self.model3 = monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features

        self.fc = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(in_features=150528, out_features=l1, bias=True),
                            nn.Dropout(0.3),
                            nn.Linear(in_features=l1, out_features=l2, bias=True),
                            nn.Dropout(0.3),
                            nn.Linear(in_features=l2, out_features=2, bias=True),
                        )
        
    def forward(self, x1, x2, x3):
        x1 = (self.model(x1))
        x2 = (self.model2(x2))
        x3 = (self.model3(x3))

        combined = torch.cat((x1, x2, x3),dim=1)
        combined = combined.reshape(combined.size(0), -1)
        logits = self.fc(combined) 
 
        return logits


def get_f1_score_precision_recall_three_inputs(model, dataloader):
    truth = []
    predections = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            model.eval()
            input_1 = batch["image"][0].to("cuda", dtype=torch.float)
            input_2 = batch["image"][1].to("cuda", dtype=torch.float)
            input_3 = batch["image"][2].to("cuda", dtype=torch.float)
        
        
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to("cuda")

            outputs = model.to("cuda")(input_1,input_2,input_3)
        
            m = nn.Softmax()
            softmax_output = m(outputs)       
            #outputs = torch.column_stack((softmax_output[:,0] > 0.110333, softmax_output[:,1])).float()

            if len(predections) == 0:
                predections = softmax_output.cpu().detach().numpy()
            else:
                predections= np.concatenate((predections, softmax_output.cpu().detach().numpy()), axis=0)
            
            for k in range(labels.shape[0]):
                truth.append(labels[k].item())

    predicted_classes = np.argmax(predections, axis=1)
    tp = 0
    fn = 0
    fp = 0
    for i in range(len(truth)):

        if (truth[i]==1 and predicted_classes[i] ==1):

            tp+=1

        elif (truth[i]==1 and predicted_classes[i] ==0):

            fn+=1

        elif (truth[i]==0 and predicted_classes[i] ==1):

            fp+=1

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1_score = 2 * precision * recall / (precision + recall)

    print("f1 score = {}, precision ={}, recall = {}".format(f1_score, precision, recall))

config = {
    "l1": tune.sample_from(lambda _: 2**np.random.randint(10, 17)),
    "l2": tune.sample_from(lambda _: 2**np.random.randint(10, 17)),
    "lr": tune.loguniform(1e-4, 1e-6, 1e-7),
    "batch_size": tune.choice([2, 4, 8, 16, 32])
}

#batch_size = int(config["batch_size"])
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)







train_acc = []
val_acc = []
train_loss = []
val_loss = []



def accuracy(outputs, labels):
#     auroc = AUROC(num_classes=3)
#     return auroc(outputs, labels)

    return (outputs == labels).float().sum()/(labels.shape[0])

def train_three_views(config):
    model = monai_densenet_three_views(config["l1"], config["l2"])

    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)


    dataset_train = FullBrainDataset(datafram=train_set,
                                 lookup_df=lookup_table,
                                 MRI_type="T2w",
                                 transform=transforms.Compose([
                                               Pad(240, 240),
                                               #Normalize(),
                                               zscore(),
                                               Normalize(),         
                                               Gaussian_Noise(0, 0.03),
                                               T.ToTensor(),
                                               T.RandomVerticalFlip(), 
                                               #T.RandomPerspective(0.2,p=0.4),
                                               T.RandomHorizontalFlip(),
                                               T.RandomRotation((0,10)), # not a huge decline
                                               T.RandomAdjustSharpness(sharpness_factor=3, p = 0.5),
                                               #T.Normalize(mean = [0.0085], std = [1.0132]),
                                                                         
                        ]))
                        
                        

    dataset_test = FullBrainDataset(datafram=test_set,
                                lookup_df=lookup_table,
                                MRI_type="T2w",
                                transform=transforms.Compose([
                                               Pad(240, 240),
                                               
                                               zscore(),
                                               Normalize(),
                                               T.ToTensor(),
                                               #T.Normalize(mean = [0.0085], std = [1.0132]),                           
                        ]))

    batch_size = int(config["batch_size"])


        
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size,
                            num_workers=28, shuffle=True, collate_fn=collate_fn)

    val_dataloader = DataLoader(dataset_test, batch_size=batch_size,
                            num_workers=28, shuffle=False, collate_fn=collate_fn)
    checkpoint_dir = ("./ray_checkpoint")
    best_acc1 = 0
    plot=False
    num_epochs = 150
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        if epoch % 5 == 0:
               print('Epoch {}/{}'.format(epoch, num_epochs - 1))
               print('-' * 10)
        
        train_running_loss = 0.0
        train_running_corrects = 0.0
        
        val_running_loss = 0.0
        val_running_corrects = 0
        
        count_val_acc = 0
        count_train_acc = 0


        for i, batch in enumerate(train_dataloader):   

            model.train()
            input_1 = batch["image"][0].to(device, dtype=torch.float)
            input_2 = batch["image"][1].to(device, dtype=torch.float)
            input_3 = batch["image"][2].to(device, dtype=torch.float)
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to(device)
            #print("labels: ", labels)
            
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(input_1, input_2, input_3)
                #print(outputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
 

                loss.backward()
                optimizer.step()

            # statistics
            try:
                train_running_corrects += accuracy(preds, labels)
            except:
                pass
            else:
                count_train_acc += 1
                
            train_running_loss += loss.item()
 
        for i, batch in enumerate(val_dataloader):
            model.eval()
            input_1 = batch["image"][0].to(device, dtype=torch.float)
            input_2 = batch["image"][1].to(device, dtype=torch.float)
            input_3 = batch["image"][2].to(device, dtype=torch.float)
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to(device)
            
                            
            with torch.no_grad():
                outputs = model(input_1, input_2, input_3)
                _, preds = torch.max(outputs, 1)  
                              
                loss = criterion(outputs, labels)
            
            try:
                val_running_corrects += accuracy(preds, labels)
            except:
                pass
            else:
                count_val_acc += 1
                
            val_running_loss += loss.item()


        epoch_acc_val = val_running_corrects / len(val_dataloader)
        epoch_acc_train = train_running_corrects / len(train_dataloader)
        train_running_loss = train_running_loss / len(train_dataloader)
        val_running_loss = val_running_loss / len(val_dataloader)
        
        is_best = epoch_acc_val > best_acc1
        best_acc1 = max(epoch_acc_val, best_acc1)

        tune.report(loss=val_running_loss, accuracy=epoch_acc_val)
        if is_best:
            best_model = copy.deepcopy(model.state_dict())
            #torch.save(model,"./best_model.pt" )
            print("best val acc = ", best_acc1)
        
        try:      
            train_acc.append(epoch_acc_train.item())
            
        except:
            train_acc.append(epoch_acc_train)
            pass
        
        try:
            val_acc.append(epoch_acc_val.item())
            
        except:
            val_acc.append(epoch_acc_val)
            pass
        
        try:
            train_loss.append(train_running_loss.item())
            
        except:
            train_loss.append(train_running_loss)
            pass
        
        try:
            val_loss.append(val_running_loss.item())
        except:
            val_loss.append(val_running_loss)
            pass
        
#         print('val acc: {:.4f}'.format(epoch_acc_val))
#         print('train acc: {:.4f}'.format(epoch_acc_train))

#         print('train loss: {:.4f}'.format(train_running_loss))
#         print('val loss: {:.4f}'.format(val_running_loss))

        if plot:
            plot_performance(val_loss, train_loss, val_acc, train_acc)

    print()

    with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)


    #torch.save(model.state_dict(), save_path)
    return model, best_model



scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=150,
        grace_period=1,
        reduction_factor=2)

reporter = CLIReporter(
        parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

result = tune.run(
    partial(train_three_views),
    config=config,
    resources_per_trial={"gpu": 2},
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end=True)


best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))

best_trained_model = monai_densenet_three_views(best_trial.config["l1"], best_trial.config["l2"])
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if 2 > 1:
        best_trained_model = nn.DataParallel(best_trained_model)
best_trained_model.to(device)

best_checkpoint_dir = best_trial.checkpoint.value
model_state, optimizer_state = torch.load(os.path.join(
    best_checkpoint_dir, "checkpoint"))
best_trained_model.load_state_dict(model_state)

dataset_test = FullBrainDataset(datafram=test_set,
                                lookup_df=lookup_table,
                                MRI_type="T2w",
                                transform=transforms.Compose([
                                               Pad(240, 240),
                                               
                                               zscore(),
                                               Normalize(),
                                               T.ToTensor(),
                                               #T.Normalize(mean = [0.0085], std = [1.0132]),                           
                        ]))

dataloader_test = DataLoader(dataset_test, batch_size=16,
                         num_workers=28, shuffle=False, collate_fn=collate_fn)
get_f1_score_precision_recall_three_inputs(best_trained_model,dataloader_test)


