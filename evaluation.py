from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import classification_report

#evaluation metrics for a three-inputs experiment
def get_confusion_matrix_AUC_three_inputs(model, dataloader, nb_classes=2):
    truth = []
    predections = []
    positives = []
    negatives = []

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
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

            preds = torch.argmax(outputs,1)
            if len(predections) == 0:
                predections = softmax_output.cpu().detach().numpy()
            else:
                predections= np.concatenate((predections, softmax_output.cpu().detach().numpy()), axis=0)
            
            for k in range(labels.shape[0]):
                truth.append(labels[k].item())


            for t, p in zip(labels.view(-1), preds.view(-1)): 
                confusion_matrix[1-t.long(),1-p.long()] += 1

    print(confusion_matrix)


    return roc_auc_score(truth , predections[:,1]) 

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
            #softmax_output = m(outputs)       
            #outputs = torch.column_stack((softmax_output[:,0] > 0.110333, softmax_output[:,1])).float()

            if len(predections) == 0:
                predections = outputs.cpu().detach().numpy()
            else:
                predections= np.concatenate((predections, outputs.cpu().detach().numpy()), axis=0)
            
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

    try:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    except:
        print("tp = ", tp)
        print("fp = ", fp)
        print("fn = ", fn)
        print("p = ", precision)
        print("r = ", recall)
        f1_score = 0

    print("f1 score = {}, precision ={}, recall = {}".format(f1_score, precision, recall))

    return f1_score, recall, precision





#evaluation metrics for a six-inputs experiment
def get_f1_score_precision_recall_six_inputs(model, dataloader):
    truth = []
    predections = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            model.eval()
            input_1 = batch["image"][0].to("cuda", dtype=torch.float)
            input_2 = batch["image"][1].to("cuda", dtype=torch.float)
            input_3 = batch["image"][2].to("cuda", dtype=torch.float)
            input_4 = batch["image"][3].to("cuda", dtype=torch.float)
            input_5 = batch["image"][4].to("cuda", dtype=torch.float)
            input_6 = batch["image"][5].to("cuda", dtype=torch.float)
        
        
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to("cuda")

            outputs = model.to("cuda")(input_1, input_2, input_3, input_4, input_5, input_6)
        
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

def get_confusion_matrix_AUC_six_inputs(model, dataloader, nb_classes=2):
    truth = []
    predections = []

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            model.eval()
            input_1 = batch["image"][0].to("cuda", dtype=torch.float)
            input_2 = batch["image"][1].to("cuda", dtype=torch.float)
            input_3 = batch["image"][2].to("cuda", dtype=torch.float)
            input_4 = batch["image"][3].to("cuda", dtype=torch.float)
            input_5 = batch["image"][4].to("cuda", dtype=torch.float)
            input_6 = batch["image"][5].to("cuda", dtype=torch.float)
        
        
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to("cuda")

            outputs = model.to("cuda")(input_1, input_2, input_3, input_4, input_5, input_6)
        
            m = nn.Softmax()
            softmax_output = m(outputs)       
            #outputs = torch.column_stack((softmax_output[:,0] > 0.110333, softmax_output[:,1])).float()

            preds = torch.argmax(outputs,1)
            if len(predections) == 0:
                predections = softmax_output.cpu().detach().numpy()
            else:
                predections= np.concatenate((predections, softmax_output.cpu().detach().numpy()), axis=0)
            
            for k in range(labels.shape[0]):
                truth.append(labels[k].item())
            
            for t, p in zip(labels.view(-1), preds.view(-1)):           
                confusion_matrix[1-t.long(),1-p.long()] += 1

    print(confusion_matrix)


    y = label_binarize(truth, classes=[0,1])


    pr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr = dict()
    tpr = dict()
    thresholds = dict()

    for i in range(nb_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y[:,i], predections[:,i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])

  
    for i in range(nb_classes):
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


#evaluation metrics for a two-inputs experiment
def get_confusion_matrix_AUC_two_inputs(model, dataloader, input1, input2,  nb_classes=2):
    truth = []
    predections = []

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            model.eval()
            input_1 = batch["image"][input1].to("cuda", dtype=torch.float)
            input_2 = batch["image"][input2].to("cuda", dtype=torch.float)

        
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to("cuda")

            outputs = model.to("cuda")(input_1, input_2)
        
            m = nn.Softmax()
            softmax_output = m(outputs)       
            #outputs = torch.column_stack((softmax_output[:,0] > 0.110333, softmax_output[:,1])).float()

            preds = torch.argmax(outputs,1)
            if len(predections) == 0:
                predections = softmax_output.cpu().detach().numpy()
            else:
                predections= np.concatenate((predections, softmax_output.cpu().detach().numpy()), axis=0)
            
            for k in range(labels.shape[0]):
                truth.append(labels[k].item())
            
            for t, p in zip(labels.view(-1), preds.view(-1)):           
                confusion_matrix[1-t.long(),1-p.long()] += 1

    print(confusion_matrix)


    y = label_binarize(truth, classes=[0,1])


    pr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr = dict()
    tpr = dict()
    thresholds = dict()

    for i in range(nb_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y[:,i], predections[:,i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])

  
    for i in range(nb_classes):
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




def get_f1_score_precision_recall_two_inputs(model, dataloader, input1, input2):
    truth = []
    predections = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            model.eval()
            input_1 = batch["image"][input1].to("cuda", dtype=torch.float)
            input_2 = batch["image"][input2].to("cuda", dtype=torch.float)
        
        
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to("cuda")

            outputs = model.to("cuda")(input_1,input_2)
        
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


#evaluation metrics for a single-inputs experiment

def get_confusion_matrix_AUC_single_input(model, dataloader, axis, nb_classes=2):
    truth = []
    predections = []

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            model.eval()
            
            input = batch["image"][axis].to("cuda", dtype=torch.float)
            if (len(input.shape)) != 4:
                input = batch["image"].to("cuda", dtype=torch.float)
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to("cuda")
            outputs = model.to("cuda")(input)
        
            m = nn.Softmax()
            softmax_output = m(outputs)       
            #outputs = torch.column_stack((softmax_output[:,0] > 0.110333, softmax_output[:,1])).float()

            preds = torch.argmax(outputs,1)
            if len(predections) == 0:
                predections = softmax_output.cpu().detach().numpy()
            else:
                predections= np.concatenate((predections, softmax_output.cpu().detach().numpy()), axis=0)
            
            for k in range(labels.shape[0]):
                truth.append(labels[k].item())
            
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[1-t.long(),1-p.long()] += 1

    print(confusion_matrix)


    

    y = label_binarize(truth, classes=[0,1])



    pr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr = dict()
    tpr = dict()
    thresholds = dict()

    print(y.shape)
    print(predections.shape)
    for i in range(nb_classes-1):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y[:,i], predections[:,i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])

    print(roc_auc)

     

    for i in range(nb_classes-1):
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

    return roc_auc_score(truth , predections[:,1])

def get_f1_score_precision_recall_single_input(model, dataloader, axis):
    truth = []
    predections = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            model.eval()
            
            input = batch["image"][axis].to("cuda", dtype=torch.float)
            if (len(input.shape)) != 4:
                input = batch["image"].to("cuda", dtype=torch.float)

            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to("cuda")

            outputs = model.to("cuda")(input)

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


    print(classification_report(truth,predicted_classes))

    print("f1 score = {}, precision ={}, recall = {}".format(f1_score, precision, recall))
    return f1_score, recall, precision



def accuracy_on_patient_level(model, dataloader, df_test):

    # first find the truth values and predicted values for each slide
    truth = []
    predections = []
    all_patient_predictions = []
    all_patient_truth = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            model.eval()
            input = batch["image"].to("cuda", dtype=torch.float)
            labels = batch["label"]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels.astype('long')).to("cuda")

            outputs = model.to("cuda")(input)
        
            m = nn.Softmax()
            softmax_output = m(outputs)       
            #outputs = torch.column_stack((softmax_output[:,0] > 0.110333, softmax_output[:,1])).float()

            if len(predections) == 0:
                predections = softmax_output.cpu().detach().numpy()
            else:
                predections= np.concatenate((predections, softmax_output.cpu().detach().numpy()), axis=0)
            
            for k in range(labels.shape[0]):
                truth.append(labels[k].item())

    #find confusion matrix
    confusion_matrix = torch.zeros(2, 2)

    #now find the confusion matrix for each patient
    predections = np.argmax(predections, axis=1)
    group_by_patient = df_test.groupby("BraTS21ID")

    count = 0
    for i, group in group_by_patient:
        count += 1
        patient_truth = group["MGMT_value"].to_numpy().astype(np.int64)
        try:
            _ = [patient_truth[patient_truth != 2][0]]
        except:
            continue

        patient_preds = predections[0:len(patient_truth)].astype(np.int64)
        predections = predections[len(patient_truth):]
        patient_truth = [patient_truth[patient_truth != 2][0]]



        try:
            patient_preds = [np.bincount(patient_preds[patient_preds!=2]).argmax()]
        except:
            print(patient_preds)
            print([int(not (patient_truth[0]))])
            patient_preds = [int(not (patient_truth[0]))]

        all_patient_truth.append(patient_truth[0])

        print(len(all_patient_truth))

        all_patient_predictions.append(patient_preds[0])

        for t, p in zip(patient_truth, patient_preds):           
                confusion_matrix[1-t,1-p] += 1

    print(confusion_matrix)


    #now find the ROC curves for each patient
    # y = label_binarize(all_patient_truth, classes=[0,1])
    # lb = preprocessing.LabelBinarizer()
    # lb.fit([0,1])
    # p = lb.transform(all_patient_predictions)

    # n_classes = 1

    # pr = dict()
    # tpr = dict()
    # roc_auc = dict()

    # fpr = dict()
    # tpr = dict()
    # thresholds = dict()


    # for i in range(n_classes):
    #     fpr[i], tpr[i], thresholds[i] = roc_curve(y[:,i], p[:,i], drop_intermediate=False)
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    y = label_binarize(all_patient_predictions, classes=[0,1,2])
    print(y.shape)
    print(y[:,1].shape)
    auc =  roc_auc_score(all_patient_truth , y[:,1]) 

  
    # for i in range(n_classes):
    #     plt.figure()
    #     plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.5f)' % roc_auc[i])
    #     plt.plot([0, 1], [0, 1], 'k--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Class ' + str(i))
    #     plt.legend(loc="lower right")
    #     plt.show()


    tp = 0
    fn = 0
    fp = 0
    for i in range(len(all_patient_predictions)):

        if (all_patient_truth[i]==1 and all_patient_predictions[i] ==1):

            tp+=1

        elif (all_patient_truth[i]==1 and all_patient_predictions[i] ==0):

            fn+=1

        elif (all_patient_truth[i]==0 and all_patient_predictions[i] ==1):

            fp+=1

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1_score = 2 * precision * recall / (precision + recall)

    print("auc = {},f1 score = {}, precision ={}, recall = {}".format(auc, f1_score, precision, recall))


    
    