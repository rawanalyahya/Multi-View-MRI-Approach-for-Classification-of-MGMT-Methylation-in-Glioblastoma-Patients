import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import monai
# train_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/slice_labeling_t2_train.csv")
# test_set = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/slice_labeling_t2_test.csv")

# patient_num = []
# for i in range(test_set.shape[0]):
#      patient_num.append(test_set.iloc[i].slice_id.split("_")[1])

# test_set["patient_number"] = patient_num
# test_set.to_csv("slice_labeling_t2_test_with_patient_label.csv")

# image = nib.load("/home/rfyahya/Desktop/Rawan/brain/clean/preprocessed/train/00000/T2w/T2_to_SRI.nii.gz").get_fdata()[:,:,69]
# plt.imsave("./brain3.png",image,cmap="gray")

print(monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2))