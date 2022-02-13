from torch.utils.data import Dataset
import nibabel as nib
import torch
import nibabel as nib

#this dataset class finds for a slice by slice labeling approach
class FullBrainSlicesDataset(Dataset):

    def __init__(self, datafram, transform=None):
        """
        Args:
            datafram (pandas.dataframe): dataframe of dataset with labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = datafram
        self.transform = transform

    def __len__(self):
        
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = nib.load(self.df.iloc[idx]["path"]).get_fdata()[:,:,self.df.iloc[idx]["slice_number"]]
             
        label = self.df["MGMT_value"][int(idx)]
        sample = {'image':image, 'label': label}
        
        
        if self.transform:
            for i in range(1):
                sample['image']= self.transform(sample['image'].copy())
  
        #sample['image'] = self.to_tensor(sample['image'])
        return sample