import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import torch.optim as optim
import sys, os
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("/home/rfyahya/Desktop/Rawan/brain/clean/visual studios/")
from datasets.slice_by_slice import FullBrainSlicesDataset
from brain_classification_preperation import *
from my_models.slice_by_slice_resnet50 import resnet50_slices
from evaluation import *

train_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/slice_labeling_t2_train.csv")
test_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/visual studios/slice_labeling_t2_test_with_patient_label.csv")


dataset_train = FullBrainSlicesDataset(datafram=train_set,
                            transform=transforms.Compose([
                                               Pad(240, 240),
                                               Normalize(),
                                               Gaussian_Noise(0, 0.01),
                                               T.ToTensor(),
                                               T.RandomVerticalFlip(), 
                                               #T.RandomPerspective(0.2,p=0.4),
                                               T.RandomHorizontalFlip(),
                                               T.RandomRotation((0,10)),
                                               T.RandomAdjustSharpness(sharpness_factor=3, p = 0.5)  ,  
                                               #T.Normalize(mean = [0.2668], std = [0.5038]),                           
                        ]))
                        
                        

dataset_test = FullBrainSlicesDataset(datafram=test_set,
                            transform=transforms.Compose([
                                               Pad(240, 240),
                                               Normalize(),
                                               T.ToTensor(),
                                               #T.Normalize(mean = [0.2668], std = [0.5038]),                           
                        ]))



device = ("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
model = resnet50_slices()
model = model.to(device)
if torch.cuda.device_count() > 1 and device == 'cuda':
       model = nn.DataParallel(model)

model = torch.load("./best_model_slice_by_slice.pt")

batch_size = 32
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                         num_workers=20, shuffle=True, collate_fn=collate_fn)

dataloader_test = DataLoader(dataset_test, batch_size=batch_size,
                         num_workers=20, shuffle=False, collate_fn=collate_fn)

train_acc = []
val_acc = []
train_loss = []
val_loss = []

criterion = nn.CrossEntropyLoss()#weight=torch.from_numpy(weight).float().to(device))
lr =  0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=0.1)
best_model = -1
best_acc1 = -1
        
        
def accuracy(outputs, labels):
#     auroc = AUROC(num_classes=3)
#     return auroc(outputs, labels)

    return (outputs == labels).float().sum()/(labels.shape[0])

def train_slices(num_epochs, train_dataloader, val_dataloader, plot=False):
    global best_model
    global best_acc1

    best_acc1 = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
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
            input = batch["image"].to(device, dtype=torch.float)
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to(device)
            #print("labels: ", labels)
            
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(input)
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
            input = batch["image"].to(device, dtype=torch.float)
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to(device)
            
                            
            with torch.no_grad():
                outputs = model(input)
                _, preds = torch.max(outputs, 1)  
                              
                loss = criterion(outputs, labels)
            
            try:
                val_running_corrects += accuracy(preds, labels)
            except:
                pass
            else:
                count_val_acc += 1
                
            val_running_loss += loss.item()
        scheduler.step()
        epoch_acc_val = val_running_corrects / len(val_dataloader)
        epoch_acc_train = train_running_corrects / len(train_dataloader)
        train_running_loss = train_running_loss / len(train_dataloader)
        val_running_loss = val_running_loss / len(val_dataloader)
        
        is_best = epoch_acc_val > best_acc1
        best_acc1 = max(epoch_acc_val, best_acc1)
        if is_best:
            best_model = copy.deepcopy(model.state_dict())
            torch.save(model,"./best_model_slice_by_slice50.pt" )
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


# train_slices(50, dataloader_train, dataloader_test, plot = False)
# torch.save(model,"./last_model_slice_by_slice50.pt" )
# model = resnet50_slices()
# model = torch.load("./last_model_slice_by_slice50.pt")

# # print("best accuracy for multi view = ", best_acc1)
# # #plot_performance(val_loss, train_loss, val_acc, train_acc)
# print("last model info")
# # get_confusion_matrix_AUC_single_input(model, dataloader_test,nb_classes=3)
# # get_f1_score_precision_recall_single_input(model,dataloader_test)
# accuracy_on_patient_level(model, dataloader_test, test_set)

print("\n best model info")
best_model = resnet50_slices()
best_model = torch.load("./best_model_slice_by_slice.pt")
# # get_confusion_matrix_AUC_single_input(best_model, dataloader_test,nb_classes=3)
# # get_f1_score_precision_recall_single_input(best_model,dataloader_test)
accuracy_on_patient_level(best_model, dataloader_test, test_set)

