from torch.utils.data import Dataset
import nibabel as nib
import torch
import pandas as pd

pd.options.display.max_colwidth = 100
#this dataset class finds the largest tumor without using lookup table
class FullBrainDataset(Dataset):

    def __init__(self, datafram, mask_df, MRI_type, transform=None):
        """
        Args:
            datafram (pandas.dataframe): dataframe of dataset with labels
            mask_df (pandas.dataframe): dataframe of masks
            MRI_type: which MRI type is this [FLAIR, T1wCE, T2w, T1w]
            transform: if passed apply transformation to input images
        """
       
        self.df = datafram
        self.transform = transform
        self.mask_df = mask_df
        self.MRI_type = MRI_type

        assert self.MRI_type in ["FLAIR", "T1wCE", "T2w", "T1w"]

    def __len__(self):
        return self.df.groupby("BraTS21ID").count().shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        img_name = self.df[(self.df["BraTS21ID"] == self.df.iloc[idx*4]["BraTS21ID"]) & (self.df["mri_type"] == self.MRI_type)]["path"].values[0]
        mask_name = self.mask_df["path"][int(idx)]
        mask = nib.load(mask_name).get_fdata()
        largest_slice_d, largest_slice_w, largest_slice_h = find_largest_tumor(mask)
        

        img_d  = nib.load(img_name).get_fdata()[:,:,largest_slice_d]
        img_w  = nib.load(img_name).get_fdata()[:,largest_slice_w,:]
        img_h  = nib.load(img_name).get_fdata()[largest_slice_h,:,:]
          
        if img_d .max() == 0 or img_w.max() == 0 or img_h.max() == 0:
            return None
        
        label = self.df["MGMT_value"][int(idx)*4]
        sample = {'image':[img_h,img_w,img_d], 'label': label}
        
        if self.transform:
            for i in range(3):
                sample['image'][i]= self.transform(sample['image'][i].copy())
                
        return sample