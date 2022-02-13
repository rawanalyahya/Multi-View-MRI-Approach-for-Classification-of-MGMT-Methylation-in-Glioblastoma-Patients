from unittest import TestLoader
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
from datasets.with_look_up_table import FullBrainDataset
from brain_classification_preperation import *
from my_models.monai_densenet import monai_densenet_three_views
from evaluation import *
from sklearn.model_selection import KFold


#read csv files
train_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/train_df.csv")
test_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/test_df.csv")
#use a look-up table to find the slices with the largest tumor without having to loop everytime 
lookup_table = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/slice_lookup.csv")
dataset_df = pd.concat([train_set, test_set]).reset_index(drop=True)
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

#define datasets 

                                        

dataset = FullBrainDataset(datafram=dataset_df,
                                lookup_df=lookup_table,
                                MRI_type="T2w",
                                transform=transforms.Compose([
                                               Pad(240, 240),
                                               
                                               zscore(),
                                               Normalize(),
                                               T.ToTensor(),
                                               #T.Normalize(mean = [0.0085], std = [1.0132]),                           
                        ]))



device = ("cuda" if torch.cuda.is_available() else "cpu")



batch_size = 16
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


torch.manual_seed(99)
torch.cuda.manual_seed_all(99)
np.random.seed(99)
random.seed(99)
torch.cuda.manual_seed(99)

model = monai_densenet_three_views()

model = model.to(device)
if torch.cuda.device_count() > 1 and device == 'cuda':
        model = nn.DataParallel(model)


train_acc = []
val_acc = []
train_loss = []
val_loss = []

criterion = nn.CrossEntropyLoss()#weight=torch.from_numpy(weight).float().to(device))
lr = 10e-7
best_model = -1
best_acc1 = -1

# mean = 0.
# std = 0.

# loader = DataLoader(dataset_train, batch_size=len(train_set),
#                          num_workers=28, shuffle=True, collate_fn=collate_fn)
# data = next(iter(loader))

# print(data["image"][0].shape)
# mean  += data["image"][0].mean()
# std +=  data["image"][0].std()
# print("mean = ", mean, " std = ", std)
# mean = 0.
# std = 0.
# for i in range(3):
#     print(i)
#     print(data["image"][i].shape)
#     mean  += data["image"][i].mean()
#     std +=  data["image"][i].std()
#     print("mean = ", mean, " std = ", std)
# mean = mean/3
# std = std/3
# print("mean = ", mean, " std = ", std)

def accuracy(outputs, labels):
#     auroc = AUROC(num_classes=3)
#     return auroc(outputs, labels)

    return (outputs == labels).float().sum()/(labels.shape[0])

def train_three_views(num_epochs, train_datasae, plot=False):
    global best_model
    global best_acc1

    best_acc1 = 0
    # For fold results
    results = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_datasae)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(train_datasae, batch_size=batch_size,
                         num_workers=28,  collate_fn=collate_fn, drop_last=True, sampler=train_subsampler)

        valloader = DataLoader(train_datasae, batch_size=batch_size,
                         num_workers=28,  collate_fn=collate_fn, sampler=test_subsampler)


        model = monai_densenet_three_views()
        model.apply(reset_weights)
        model = model.to(device)
        if torch.cuda.device_count() > 1 and device == 'cuda':
                model = nn.DataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)



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


            for i, batch in enumerate(trainloader):   

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
    
            for i, batch in enumerate(valloader):
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


            epoch_acc_val = val_running_corrects / len(valloader)
            epoch_acc_train = train_running_corrects / len(trainloader)
            train_running_loss = train_running_loss / len(trainloader)
            val_running_loss = val_running_loss / len(valloader)
            
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
        results[fold] = 100.0 * epoch_acc_val

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')

    #torch.save(model.state_dict(), save_path)
    return model, best_model



train_three_views(50,dataset, plot = False)
print("best accuracy for multi view = ", best_acc1)
plot_performance(val_loss, train_loss, val_acc, train_acc)
print("last model info")

get_f1_score_precision_recall_three_inputs(model,dataloader_test)
get_confusion_matrix_AUC_three_inputs(model, dataloader_test, nb_classes=2)


print("\n best model info")
best_model = monai_densenet_three_views()
best_model = torch.load("./best_model.pt")
get_confusion_matrix_AUC_three_inputs(best_model, dataloader_test,nb_classes=2)
get_f1_score_precision_recall_three_inputs(best_model,dataloader_test)
