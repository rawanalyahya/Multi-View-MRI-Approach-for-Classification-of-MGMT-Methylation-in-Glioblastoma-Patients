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
from datasets.multi_mri import FullBrainDataset_multi_mri
from brain_classification_preperation import *
from my_models.monai_densenet import monai_densenet_three_views, multi_mri_fc_layers
from evaluation import *


#read csv files
train_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/training_with_val_split.csv")
test_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/test_df.csv")
val_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/val_set.csv")
#use a look-up table to find the slices with the largest tumor without having to loop everytime 
lookup_table = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/slice_lookup.csv")


#define datasets 
dataset_train = FullBrainDataset_multi_mri(datafram=train_set,
                                 lookup_df=lookup_table,
                                 MRI_type1="T2w",
                                 MRI_type2="T1w",
                                 transform=transforms.Compose([
                                               Pad(240, 240),
                                               zscore(),
                                               Gaussian_Noise(0, 0.03),
                                               T.ToTensor(),
                                               #T.RandomVerticalFlip(), 
                                               #T.RandomPerspective(0.2,p=0.4),
                                               #T.RandomHorizontalFlip(),
                                               #T.RandomRotation((0,10)), # not a huge decline
                                               T.RandomAdjustSharpness(sharpness_factor=3, p = 0.5),
                                               #T.Normalize(mean = [0.0085], std = [1.0132]),                           
                        ]))
                        
                        

dataset_test = FullBrainDataset_multi_mri(datafram=test_set,
                                lookup_df=lookup_table,
                                MRI_type1="T2w",
                                MRI_type2="T1w",
                                transform=transforms.Compose([
                                               Pad(240, 240),
                                               zscore(),
                                               T.ToTensor(),
                                               #T.Normalize(mean = [0.0085], std = [1.0132]),                           
                        ]))

dataset_val = FullBrainDataset_multi_mri(datafram=val_set,
                                lookup_df=lookup_table,
                                MRI_type1="T2w",
                                MRI_type2="T1w",
                                transform=transforms.Compose([
                                               Pad(240, 240),
                                               zscore(),
                                               T.ToTensor(),
                                               #T.Normalize(mean = [0.0085], std = [1.0132]),                           
                        ]))


device = ("cuda" if torch.cuda.is_available() else "cpu")



batch_size = 8
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


torch.manual_seed(99)
torch.cuda.manual_seed_all(99)
np.random.seed(99)
random.seed(99)
torch.cuda.manual_seed(99)

model = multi_mri_fc_layers()
model = model.to(device)
# if torch.cuda.device_count() > 1 and device == 'cuda':
#        model = nn.DataParallel(model)


dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                         num_workers=28, shuffle=True, collate_fn=collate_fn)

dataloader_test = DataLoader(dataset_test, batch_size=batch_size,
                         num_workers=28, shuffle=True, collate_fn=collate_fn)

dataloader_val = DataLoader(dataset_val, batch_size=batch_size,
                         num_workers=28, shuffle=True, collate_fn=collate_fn)

train_acc = []
val_acc = []
train_loss = []
val_loss = []

criterion = nn.CrossEntropyLoss()#weight=torch.from_numpy(weight).float().to(device))
lr = 10e-7
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
best_model = -1
best_acc1 = -1



def accuracy(outputs, labels):
#     auroc = AUROC(num_classes=3)
#     return auroc(outputs, labels)

    return (outputs == labels).float().sum()/(labels.shape[0])

def train_multi_mri(num_epochs, train_dataloader, val_dataloader, plot=False):
    global best_model
    global best_acc1

    best_acc1 = 0
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
            input_4 = batch["image"][3].to(device, dtype=torch.float)
            input_5 = batch["image"][4].to(device, dtype=torch.float)
            input_6 = batch["image"][5].to(device, dtype=torch.float)
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to(device)
            #print("labels: ", labels)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(input_1, input_2, input_3, input_4, input_5, input_6)
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
            input_4 = batch["image"][3].to(device, dtype=torch.float)
            input_5 = batch["image"][4].to(device, dtype=torch.float)
            input_6 = batch["image"][5].to(device, dtype=torch.float)
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to(device)
            
                            
            with torch.no_grad():
                outputs = model(input_1, input_2, input_3, input_4, input_5, input_6)
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
        if is_best:
            best_model = copy.deepcopy(model.state_dict())
            torch.save(model,"./best_model.pt" )
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


    #torch.save(model.state_dict(), save_path)
    return model, best_model


train_multi_mri(5, dataloader_train, dataloader_val, plot = False)
print("best accuracy for multi view = ", best_acc1)
plot_performance(val_loss, train_loss, val_acc, train_acc)
print("last model info")

get_f1_score_precision_recall_six_inputs(model,dataloader_test)
get_confusion_matrix_AUC_six_inputs(model, dataloader_test, nb_classes=2)

print("\n best model info")
best_model = monai_densenet_three_views()
best_model = torch.load("./best_model.pt")
get_confusion_matrix_AUC_six_inputs(best_model, dataloader_test,nb_classes=2)
get_f1_score_precision_recall_six_inputs(best_model,dataloader_test)
