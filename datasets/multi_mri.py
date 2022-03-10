from torch.utils.data import Dataset
import nibabel as nib
import torch
import nibabel as nib

#this dataset class finds the largest tumor without using lookup table
class FullBrainDataset_multi_mri(Dataset):

    def __init__(self, datafram, lookup_df, MRI_type1, MRI_type2, transform=None):
        """
        Args:
            datafram (pandas.dataframe): dataframe of dataset with labels
            lookup_df (pandas.dataframe): dataframe of look-up table
            MRI_type: which MRI type is this [FLAIR, T1wCE, T2w, T1w]
            transform: if passed apply transformation to input images
        """
       
        self.df = datafram
        self.transform = transform
        self.lookup_df = lookup_df
        self.MRI_type1 = MRI_type1
        self.MRI_type2 = MRI_type2


        assert self.MRI_type1 in ["FLAIR", "T1wCE", "T2w", "T1w"]
        assert self.MRI_type2 in ["FLAIR", "T1wCE", "T2w", "T1w"]


    def __len__(self):
        
        return self.df.groupby("BraTS21ID").count().shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name1 = self.df[(self.df["BraTS21ID"] == self.df.iloc[idx*4]["BraTS21ID"]) & (self.df["mri_type"] == self.MRI_type1)]["path"].values[0]
        img_name2 = self.df[(self.df["BraTS21ID"] == self.df.iloc[idx*4]["BraTS21ID"]) & (self.df["mri_type"] == self.MRI_type2)]["path"].values[0]


        slice = self.lookup_df[self.lookup_df["id"] == self.df.iloc[idx*4]["BraTS21ID"]]

        largest_slice_x = int(slice["x"])
        largest_slice_y = int(slice["y"])
        largest_slice_z = int(slice["z"])
        
        img_d1  = nib.load(img_name1).get_fdata()[:,:,largest_slice_z]     
        img_w1  = nib.load(img_name1).get_fdata()[:,largest_slice_y,:]
        img_h1  = nib.load(img_name1).get_fdata()[largest_slice_x,:,:]

        img_d2  = nib.load(img_name2).get_fdata()[:,:,largest_slice_z]     
        img_w2  = nib.load(img_name2).get_fdata()[:,largest_slice_y,:]
        img_h2  = nib.load(img_name2).get_fdata()[largest_slice_x,:,:]        
        
        if img_d1.max() == 0 or img_w1.max() == 0 or img_h1.max() == 0 or img_d2.max() == 0 or img_w2.max() == 0 or img_h2.max() == 0:
            return None
        
        label = self.df["MGMT_value"][int(idx)*4]

        sample = {'image':[img_h1, img_w1, img_d1, img_h2,img_w2,img_d2], 'label': label}
        
        if self.transform:
            for i in range(6):
                sample['image'][i]= self.transform(sample['image'][i].copy())
                
  
        return sample