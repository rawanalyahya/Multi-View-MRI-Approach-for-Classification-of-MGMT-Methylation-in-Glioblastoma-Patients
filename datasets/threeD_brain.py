from torch.utils.data import Dataset
import nibabel as nib
import torch
import nibabel as nib
from torchvision import transforms

#this dataset class finds the largest tumor using lookup table
class FullBrainDataset(Dataset):

    def __init__(self, local_path, datafram, MRI_type, transform=None):
        """
        Args:
            datafram (pandas.dataframe): dataframe of dataset with labels
            MRI_type: which MRI type is this [FLAIR, T1wCE, T2w, T1w]
            transform: if passed apply transformation to input images
        """
       
        self.df = datafram
        self.transform = transform
        self.local_path = local_path
        self.MRI_type = MRI_type
        assert self.MRI_type in ["FLAIR", "T1wCE", "T2w", "T1w"]

    def __len__(self):
        
        return self.df.groupby("BraTS21ID").count().shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.local_path +self.df[(self.df["BraTS21ID"] == self.df.iloc[idx]["BraTS21ID"]) & (self.df["mri_type"] == self.MRI_type)]["path"].values[0]

        brain_img  = nib.load(img_name).get_fdata()
        
          
        if brain_img.max() == 0 :
            print(self.df.iloc[idx]["BraTS21ID"])
            return None
        
        label = self.df["MGMT_value"][int(idx)]

        sample = {'image':brain_img, 'label': label}
        
        if self.transform:
               trans = transforms.Compose([transforms.ToTensor()])
               sample['image'] = trans(sample['image'].copy())
               sample['image'] = sample['image'].unsqueeze(0)
               sample['image']= self.transform(sample['image'])
                
  
        return sample