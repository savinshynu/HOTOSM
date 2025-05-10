import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class HotosmDataset(Dataset):
    """
    Needs to have 3 things:
    -- init function listing all the files
    -- a method to get each of the files 
    -- a method to get the total number of items
    """
    def __init__(self, filepath):
        self.filepath = filepath
        data = np.load(filepath)
        self.images = data['image']
        self.masks = data['mask']
        
        # Image transformations (convert to tensor), resizing and normalizing already done
        # input array needs to be converted to torch tensor format before feeding in to pytorch
        # I think this transformation converts the array to tensor, normalizes and changes the order of channels.
        #self.transform = transforms.Compose([
        #        transforms.ToTensor()])
        
    def __getitem__(self, index): # this index will accessed by the DataLoader class of pytorch. This should output each feature and label in the tensor format.
        img = self.images[index]
        mask = self.masks[index]

        """
        In this case, the images are already normalized. So they need to converted to tensors and required order change in the frequency channels.
        """

        #print(mask.dtype)
        #return img.permute(1, 2, 0), mask
        return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(mask).long()

    def __len__(self): # length of dataset
        return len(self.images)