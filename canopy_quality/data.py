import torch
from torch.utils.data import Dataset


### Create Dataloaders using the file paths ###
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create two DataLoaders, one for training and one for test
class dataset_eval(Dataset):
    def __init__(self,filelist_train, transform=None):
        """
        Args:
            filelist (string): List with all of the file paths
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.filelist = filelist_train           

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()    
            
        # Generate data
        dataset = np.load(self.filelist[idx])
        
        # X
        X=dataset[:14] # separate out the band values

        # canopy_height,tree/not tree,ndvi
        out_tree_height = dataset[14]         
        out_tree_mask = dataset[15]
        
        preds=[out_tree_height, out_tree_mask]
        
        return [X,preds]
    

