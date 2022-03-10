from torch.utils.data import Dataset
import nibabel as nib
import torch
import nibabel as nib

#this dataset class finds the largest tumor without using lookup table
class FullBrainDataset(Dataset):

    def __init__(self, local_path, datafram, lookup_df, MRI_type, transform=None):
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
        self.local_path = local_path
        self.MRI_type = MRI_type

        assert self.MRI_type in ["FLAIR", "T1wCE", "T2w", "T1w"]

    def __len__(self):
        
        return self.df.groupby("BraTS21ID").count().shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()



        img_name = self.local_path +self.df[(self.df["BraTS21ID"] == self.df.iloc[idx]["BraTS21ID"]) & (self.df["mri_type"] == self.MRI_type)]["path"].values[0]

        slice = self.lookup_df[self.lookup_df["id"] == self.df.iloc[idx]["BraTS21ID"]]

        largest_slice_x = int(slice["x"])
        largest_slice_y = int(slice["y"])
        largest_slice_z = int(slice["z"])
        
        img_d  = nib.load(img_name).get_fdata()[:,:,largest_slice_z]     
        img_w  = nib.load(img_name).get_fdata()[:,largest_slice_y,:]
        img_h  = nib.load(img_name).get_fdata()[largest_slice_x,:,:]

        
          
        if img_d.max() == 0 or img_w.max() == 0 or img_h.max() == 0:
            print(self.df.iloc[idx]["BraTS21ID"])
            return None
        
        label = self.df["MGMT_value"][int(idx)]

        

        sample = {'image':[img_h,img_w,img_d], 'label': label}
        
        if self.transform:
            for i in range(3):
                sample['image'][i]= self.transform(sample['image'][i].copy())
                
  
        return sample