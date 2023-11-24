import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


### Create Dataloaders using the file paths ###
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create two DataLoaders, one for training and one for test
class batch_data_clean(Dataset):
    def __init__(self,image_arrays, transform=None):
        """
        Args:
            image_arrays (np arrays): Ndarray with all of the images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.image_list = image_arrays           

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
            
        # Generate data
        np_image = self.image_list[idx]
        # Shape should be (1, 14, 240, 240) - add some validation
        
        np_image[np_image  < .0000001] = 0
        
        # normalize values of the input data to 0,1
        np_image = np_image/np_image.max(axis=(1),keepdims=True)
        
        np_image = torch.from_numpy(np_image)
        np_image = np_image.to(device)
        np_image = Variable(np_image.float())
        
        return np_image
