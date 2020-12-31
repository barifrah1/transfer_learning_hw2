import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random


class CIFAR10_DataLoader_Len_Limited(Dataset):
    def __init__(self, original_datset, len_limit=None):
        self.len_limit = len_limit
        self.original_datset = original_datset
        self.label_to_indices = {label: np.where(np.array(self.original_datset.targets) == label)[0].tolist()
                                 for label in [0, 1]}
        # random.shuffle(self.label_to_indices[0])
        # random.shuffle(self.label_to_indices[1])
        self.ind_0 = 0  # counter for label 0
        self.ind_1 = 0  # couner for label 1

    def __getitem__(self, ind):
        # In case we have already return more label 1 rows than label 0 rows, so the next row will be taken
        # from the minority class
        if(self.ind_0 <= self.ind_1):
            next_ind = self.label_to_indices[0][self.ind_0]
            label = self.original_datset.targets[next_ind]
            self.ind_0 += 1
            if(ind == self.len_limit-1):
                self.ind_0 = 0
                self.ind_1 = 0
            return self.original_datset[next_ind]
        else:
            next_ind = self.label_to_indices[1][self.ind_1]
            label = self.original_datset.targets[next_ind]
            self.ind_1 += 1
            if(ind == self.len_limit-1):
                self.ind_0 = 0
                self.ind_1 = 0
            return self.original_datset[next_ind]

    def __len__(self):  # return at most len_limit rows from the dataset
        # we were asked to limit the number of training examples to do the training process more efficient
        if self.len_limit:
            return min(len(self.original_datset), self.len_limit)
        return len(self.original_datset)
