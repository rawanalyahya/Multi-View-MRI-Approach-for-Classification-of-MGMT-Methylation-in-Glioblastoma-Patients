# Comparisons with SOTA methods on MGMT methylation classification

We provided the code to reproduce the results we got from running each experiment. All code was run using python version 3.9.12. 

## Compare with Kaggle winner model 

Code made available by the author at [his GitHub page](https://github.com/FirasBaba/rsna-resnet10).
We modified the code such that we do not apply cross validation as the author did. We instead use our training/validation/testin split. We also some code in validation.py to measure more metrics. 
The code expects DICOM files without preproccessing which can be downloaded from [here](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/data).

To run this experiment:

```
bash rsna-resnet10/train_valid.sh
```

## Compare with method by Korfiatis et. al [2017](https://link.springer.com/article/10.1007/s10278-017-0009-z)

This approch uses ResNet50 trained on a slice-by-slice basis assigning each slice a label. Slices containing a tumor were labeled methylated or unmethylated based on the patient's label. Slices not containing a tumor were labeled as normal. To evaluate the model on a patient level, majority voting of the methylated and unmethylated labels was applied to assign a label for each patient. 

### running the experiment:

To run the experiment run main.py using this command:
```
python main.py -n slice_by_slice -t {train/eval} -b none -lr none -opt none  -l1 none -l2 none -loss none
```
Note that the hyperparameters are passed as none since the hyperparameters in this experiment is set to the values stated in their paper.

## Compare with method by Zhi-Cheng Li [2018](https://link.springer.com/article/10.1007/s00330-017-5302-1)

This approch uses radiomics features as an input to the classifier instead of a visual MRI input. We extracted specific radiomics features from each sub-region of the tumor: edema, enhancing tumor, and necrosis using the [pyradiomics library](https://github.com/AIM-Harvard/pyradiomics). We then passed the extracted radiomics features to a random forest to get the final classification output. The validation set was used in this experiment to optimize the hyperparameters of the random forest.

To run this experiment:
- First, extract the radiomics feature by running the Jupyter Notebook ExtractRadiomics.ipynb
- Run the random forest classifier by running the Jupyter Notebook RandomForestModel.ipynb
