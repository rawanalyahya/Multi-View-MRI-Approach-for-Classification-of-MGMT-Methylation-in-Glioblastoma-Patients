import os

import monai
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import classification_report
from delong import *
from dataset import BrainRSNADataset

val_df = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/visual studios/dataframes/testing_set_t2w.csv").reset_index(drop=True)
val_index = val_df.index
targets = val_df.MGMT_value.values

device = torch.device("cpu")
model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
model.to(device)

tta_true_labels = []
tta_preds = []
preds_f = np.zeros(len(val_df))

model.load_state_dict(torch.load("/home/rfyahya/Desktop/Rawan/brain/clean/rsna-resnet10/weights/3d-resnet10_T2w_fold0_0.549.pth"))

type_ = "T2w"
fold = 0
test_dataset = BrainRSNADataset(data=val_df, is_train=True, do_load=True, ds_type=f"test_{type_}_{fold}")
test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

image_ids = []
preds = []
case_ids = []
preds_type = np.zeros(len(val_df))

truth = []
predections = []
their_preds = []
confusion_matrix = torch.zeros(2, 2)

with torch.no_grad():
    for  step, batch in enumerate(test_dl):
        model.eval()
        images = batch["image"].to(device)
        labels = batch["target"]
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels.astype('long')).to("cuda")

        outputs = model(images)

        their_preds.append(outputs.sigmoid().detach().cpu().numpy())

        m = nn.Softmax()
        #softmax_output = m(outputs)       
        #outputs = torch.column_stack((softmax_output[:,0] > 0.110333, softmax_output[:,1])).float()

        preds = outputs.sigmoid().cpu().detach().numpy() >= 0.5
        preds = preds.astype(int)
        if len(predections) == 0:
            predections = outputs.sigmoid().cpu().detach().numpy()
        else:
            predections= np.concatenate((predections, outputs.sigmoid().cpu().numpy()), axis=0)
        
        for k in range(labels.shape[0]):
            truth.append(labels[k].item())
        
        for t, p in zip(labels.view(-1), preds):
            confusion_matrix[t.long(),p] += 1


print(type(predections))

print(confusion_matrix)

y = label_binarize(truth, classes=[0,1])

pr = dict()
tpr = dict()
roc_auc = dict()

fpr = dict()
tpr = dict()
thresholds = dict()

for i in range(2-1):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y[:,i], predections, drop_intermediate=False)
    roc_auc[i] = auc(fpr[i], tpr[i])

print(predections)
print(type(predections))
print(np.array(truth).size)
print("here", predections.resize(np.array(truth).size))
auc_delong, auc_cov = delong_roc_variance(
np.array(truth),
predections)


alpha = .95
auc_std = np.sqrt(auc_cov)
lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

ci = stats.norm.ppf(
    lower_upper_q,
    loc=auc_delong,
    scale=auc_std)

ci[ci > 1] = 1

print('AUC:', auc_delong)
print('AUC COV:', auc_cov)
print('95% AUC CI:', ci)
print("AUC = " , roc_auc)

    

for i in range(2-1):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.5f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Class ' + str(i))
    plt.legend(loc="lower right")
    plt.show()

print("AUC = ", roc_auc_score(truth , predections))


predicted_classes = predections >= 0.5
predicted_classes = predicted_classes.astype(int)

tp = 0
fn = 0
fp = 0
tn = 0
for i in range(len(truth)):

    if (truth[i]==1 and predicted_classes[i] ==1):

        tp+=1

    elif (truth[i]==1 and predicted_classes[i] ==0):

        fn+=1

    elif (truth[i]==0 and predicted_classes[i] ==1):

        fp+=1

    elif (truth[i]==0 and predicted_classes[i] ==0):

        tn+=1


try:
    precision = tp / (tp+fp)
except:
    precision = 0

try:
    recall = tp / (tp+fn)
except:
    recall = 0

try:
    specificity = tn/(fp+tn)
except:
    specificity = 0
try:
    f1_score = 2 * ((precision * recall) / (precision + recall))
except:
    f1_score = 0

print("tp = ", tp)
print("fp = ", fp)
print("fn = ", fn)
print("p = ", precision)
print("r = ", recall)

print("f1 score = {}, precision ={}, recall = {}, specificity = {}".format(f1_score, precision, recall, specificity))

