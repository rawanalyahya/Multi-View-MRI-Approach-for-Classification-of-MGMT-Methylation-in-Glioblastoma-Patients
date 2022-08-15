from pickletools import optimize
from pyexpat import model
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
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("/home/rfyahya/Desktop/Rawan/brain/clean/visual studios/")
from datasets.threeD_brain import FullBrainDataset
from brain_classification_preperation import *
from my_models.monai_densenet import *
from evaluation import *
from monai.transforms import (
    AddChannel,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    EnsureType
)
import torchio as tio
import nibabel as nib

class training_3d():
    #set variables
    def __init__(self, batch_size, lr, optimizer, l1, l2, axis):
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.l1 = l1
        self.l2 = l2
        self.axis = axis
        #read csv files
        train_set = pd.read_csv("/home/rawan/rawan/dataframes/training_set_t2w.csv").reset_index(drop=True)
        train_set = train_set[(train_set["mri_type"] == "T2w")].reset_index(drop=True)
        test_set = pd.read_csv("/home/rawan/rawan/dataframes/testing_set_t2w.csv").reset_index(drop=True)
        test_set = test_set[(test_set["mri_type"] == "T2w")].reset_index(drop=True)
        val_set = pd.read_csv("/home/rawan/rawan/dataframes/validation_set_t2w.csv").reset_index(drop=True)
        val_set = val_set[(val_set["mri_type"] == "T2w")].reset_index(drop=True)


        training_transforms = tio.Compose([
            tio.transforms.CropOrPad((156, 240, 240)),
            tio.ZNormalization(masking_method = tio.ZNormalization.mean),
            tio.RescaleIntensity(out_min_max = [0,1]),
            tio.transforms.Noise(mean=0.0, std=0.03, seed = 99),
            tio.transforms.RandomFlip(axes=(0,1,2)),
            tio.RandomAffine(p= 0.5),
            #scales=(1, 1,1))
        #    degrees=10)
            ])

        test_transforms = tio.Compose([
            tio.transforms.CropOrPad((156, 240, 240)),
            tio.ZNormalization(masking_method = tio.ZNormalization.mean),
            tio.RescaleIntensity(out_min_max = [0,1]),
            ])

        #define datasets 
        dataset_train = FullBrainDataset(local_path = "../",
                                        datafram=train_set,
                                        MRI_type="T2w",
                                        transform=training_transforms)
                                                    

        dataset_test = FullBrainDataset(local_path = "../",
                                        datafram=test_set,
                                        MRI_type="T2w",
                                        transform=test_transforms)

        dataset_val = FullBrainDataset(local_path = "../",
                                        datafram=val_set,
                                        MRI_type="T2w",
                                        transform=test_transforms)




        self.device = ("cuda" if torch.cuda.is_available() else "cpu")


        batch = dataset_train[3]
        brain = batch["image"]
        print("brain_dimentions = ", brain[:,:,150].permute(1, 2, 0).squeeze().shape)
        plt.imshow(brain[:,:,150].permute(1, 2, 0).squeeze())
        plt.show()
        plt.imsave("./brain.png", brain[:,:,150].permute(1, 2, 0).squeeze() )
        print("show image")

        batch = dataset_train[11]
        brain = batch["image"]
        print("brain_dimentions = ", brain[:,:,150].permute(1, 2, 0).squeeze().shape)
        plt.imshow(brain[:,:,150].permute(1, 2, 0).squeeze())
        plt.show()
        plt.imsave("./brain2.png", brain[:,:,150].permute(1, 2, 0).squeeze() )
        print("show image")


        # ni_img = nib.Nifti1Image(brain.squeeze().cpu().detach().numpy(), affine=np.eye(4))
        # nib.save(ni_img, "3d_brain.nii")

        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch)

        torch.manual_seed(99)
        torch.cuda.manual_seed_all(99)
        np.random.seed(99)
        random.seed(99)
        torch.cuda.manual_seed(99)

        self.model = monai_densenet_3D(l1 = self.l1, l2 = self.l2)

        self.model = self.model.to(self.device)
        # if torch.cuda.device_count() > 1:
        #         self.model = nn.DataParallel(self.model)

        self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size,
                                num_workers=0, shuffle=True, collate_fn=collate_fn, drop_last=True)

        self.dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size,
                                num_workers=0, shuffle=False, collate_fn=collate_fn)

        self.dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size,
                                num_workers=0, shuffle=False, collate_fn=collate_fn)


        self.criterion = nn.CrossEntropyLoss()#weight=torch.from_numpy(weight).float().to(device))

        # for batch in  self.dataloader_train:
        #     sample_image = batch['image']    # Reshape them according to your needs.
        #      sample_label = batch['label']

        #     print("dataloader image size = ", sample_image.shape)

        assert self.optimizer in ["Adam", "SGD", "Adagrad"]

        if self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        elif self.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-7)

        elif self.optimizer == "Adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=1e-5)



    def accuracy(self, outputs, labels):
        return (outputs == labels).float().sum()/(labels.shape[0])

    def train_single_view(self, num_epochs, train_dataloader, val_dataloader, plot=False):
        best_acc1 = -1
        train_acc = []
        val_acc = []
        train_loss = []
        val_loss = []
        early_stop_epoch_point = 0


        best_acc1 = -1
        for epoch in range(num_epochs):
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
                self.model.train()
                input = batch["image"].to(self.device, dtype=torch.float)
                
                labels = batch["label"]
                labels = np.asarray(labels)
                labels = torch.from_numpy(labels.astype('long')).to(self.device)
                #print("labels: ", labels)
                
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model(input)
                    #print(outputs)
                    _, preds = torch.max(outputs, 1)

                    loss = self.criterion(outputs, labels)
    

                    loss.backward()
                    self.optimizer.step()

                # statistics
                try:
                    train_running_corrects += self.accuracy(preds, labels)
                except:
                    pass
                else:
                    count_train_acc += 1
                    
                train_running_loss += loss.item()
    
            for i, batch in enumerate(val_dataloader):
                self.model.eval()
                input = batch["image"].to(self.device, dtype=torch.float)
                labels = batch["label"]
                labels = np.asarray(labels)
                labels = torch.from_numpy(labels.astype('long')).to(self.device)
                
                                
                with torch.no_grad():
                    outputs = self.model(input)
                    _, preds = torch.max(outputs, 1)  
                                
                    loss = self.criterion(outputs, labels)
                
                try:
                    val_running_corrects += self.accuracy(preds, labels)
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
            if is_best:
                torch.save(self.model,"/home/rawan/rawan/visual studios/saved_models/best_model_3dresnet.pt" )
                print("best val acc = ", best_acc1)
                early_stop_epoch_point = epoch

            if epoch == (30 + early_stop_epoch_point):
                break


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


        #torch.save(model.state_dict(), save_path)
        return self.model, val_loss, train_loss, val_acc, train_acc


    def train(self):
        last_model, val_loss, train_loss, val_acc, train_acc = self.train_single_view(400, self.dataloader_train, self.dataloader_val, plot = False)

    def eval(self):
        print("\n best model info")
        best_model = monai_densenet_3D()
        best_model = torch.load("/home/rawan/rawan/visual studios/saved_models/best_model_3dresnet.pt")
        auc = get_confusion_matrix_AUC_single_input(best_model, self.dataloader_test,self.axis,nb_classes=2)
        get_f1_score_precision_recall_single_input(best_model, self.dataloader_test, self.axis)





