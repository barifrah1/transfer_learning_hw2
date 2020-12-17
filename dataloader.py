import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np


class CIFAR10_DataLoader_Len_Limited(Dataset):
    def __init__(self, original_datset, len_limit=None):
        self.len_limit = len_limit
        self.original_datset = original_datset
        """self.target_classes = classes
        labels = np.array(self.original_datset.targets)
        mask = list(
            map(lambda x: True if x in self.target_classes else False, labels))
        self.original_datset.data = self.original_datset[mask]
        self.train_labels = labels[mask].tolist()"""

    def __getitem__(self, ind):
        return self.original_datset[ind]

    def __len__(self):
        if self.len_limit:
            return min(len(self.original_datset), self.len_limit)
        return len(self.original_datset)
