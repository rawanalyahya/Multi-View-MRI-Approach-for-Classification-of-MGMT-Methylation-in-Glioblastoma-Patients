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
from datasets.with_look_up_table import FullBrainDataset
from brain_classification_preperation import *
from my_models.monai_densenet import monai_densenet_three_views
from evaluation import *


class train_three_view():
    #set variables
    def __init__(self, batch_size, lr, optimizer, l1, l2):
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.l1 = l1
        self.l2 = l2
        #read csv files
        train_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/training_with_val_split.csv")
        test_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/test_df.csv")
        val_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/val_set.csv")
        #use a look-up table to find the slices with the largest tumor without having to loop everytime 
        lookup_table = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/slice_lookup.csv")


        #define datasets 
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

        dataset_val = FullBrainDataset(datafram=val_set,
                                        lookup_df=lookup_table,
                                        MRI_type="T2w",
                                        transform=transforms.Compose([
                                                    Pad(240, 240),
                                                    zscore(),
                                                    Normalize(),
                                                    T.ToTensor(),
                                                    #T.Normalize(mean = [0.0085], std = [1.0132]),                           
                                ]))

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")



        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch)

        torch.manual_seed(99)
        torch.cuda.manual_seed_all(99)
        np.random.seed(99)
        random.seed(99)
        torch.cuda.manual_seed(99)

        self.model = monai_densenet_three_views(l1 = self.l1, l2 = self.l2)

        self.model = self.model.to(self.device)
        if torch.cuda.device_count() > 1 and self.device == 'cuda':
                self.model = nn.DataParallel(self.model)

        self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size,
                                num_workers=28, shuffle=True, collate_fn=collate_fn, drop_last=True)

        self.dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size,
                                num_workers=28, shuffle=False, collate_fn=collate_fn)

        self.dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size,
                                num_workers=28, shuffle=False, collate_fn=collate_fn)


        self.criterion = nn.CrossEntropyLoss()#weight=torch.from_numpy(weight).float().to(device))

        assert self.optimizer in ["Adam", "SGD", "Adagrad"]

        if self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        elif self.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        elif self.optimizer == "Adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=1e-5)


    


    def accuracy(self, outputs, labels):
        return (outputs == labels).float().sum()/(labels.shape[0])

    def train_three_views(self, num_epochs, train_dataloader, val_dataloader, plot=False):
        best_acc1 = -1
        train_acc = []
        val_acc = []
        train_loss = []
        val_loss = []
        early_stop_epoch_point = 0


        best_acc1 = -1
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
                self.model.train()
                input_1 = batch["image"][0].to(self.device, dtype=torch.float)
                input_2 = batch["image"][1].to(self.device, dtype=torch.float)
                input_3 = batch["image"][2].to(self.device, dtype=torch.float)
                labels = batch["label"]
                labels = np.asarray(labels)
                labels = torch.from_numpy(labels.astype('long')).to(self.device)
                #print("labels: ", labels)
                
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model(input_1, input_2, input_3)
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
                input_1 = batch["image"][0].to(self.device, dtype=torch.float)
                input_2 = batch["image"][1].to(self.device, dtype=torch.float)
                input_3 = batch["image"][2].to(self.device, dtype=torch.float)
                labels = batch["label"]
                labels = np.asarray(labels)
                labels = torch.from_numpy(labels.astype('long')).to(self.device)
                
                                
                with torch.no_grad():
                    outputs = self.model(input_1, input_2, input_3)
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
                torch.save(self.model,"/home/rfyahya/Desktop/Rawan/brain/clean/visual studios/training/best_model.pt" )
                print("best val acc = ", best_acc1)
                early_stop_epoch_point = epoch

            if epoch == (100 + early_stop_epoch_point):
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


    def train_evaluate(self):
        last_model, val_loss, train_loss, val_acc, train_acc = self.train_three_views(400, self.dataloader_train, self.dataloader_val, plot = False)
        #plot_performance(val_loss, train_loss, val_acc, train_acc)
        print("last model info")

        get_f1_score_precision_recall_three_inputs(last_model,self.dataloader_test)
        get_confusion_matrix_AUC_three_inputs(last_model, self.dataloader_test, nb_classes=2)


        print("\n best model info")
        best_model = monai_densenet_three_views()
        best_model = torch.load("/home/rfyahya/Desktop/Rawan/brain/clean/visual studios/training/best_model.pt")
        auc = get_confusion_matrix_AUC_three_inputs(best_model, self.dataloader_test,nb_classes=2)
        f1_score, recall, precision = get_f1_score_precision_recall_three_inputs(best_model, self.dataloader_test)

        #write results in csv file
        experiments = pd.read_csv("experiments_three_view_t2w.csv")
        print(experiments)
        new_results = pd.DataFrame()
        new_results["AUC"] = [auc]
        new_results["F1 Score"] = [f1_score]
        new_results["recall"] = [recall]
        new_results["precision"]= [precision]
        new_results["lr"]= [self.lr]
        new_results["batch_size"] = [self.batch_size]
        new_results["optimizer"] = [self.optimizer]
        new_results["l1"] = [self.l1]
        new_results["l2"] = [self.l2]

        print(new_results)
        experiments = pd.concat([experiments, new_results]).reset_index(drop=True)
        experiments.to_csv("experiments_three_view_t2w.csv", index=False)

