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
        random.shuffle(self.label_to_indices[0])
        random.shuffle(self.label_to_indices[1])
        self.ind_0 = 0
        self.ind_1 = 0

    def __getitem__(self, ind):
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
            """oldLabel = self.original_datset.targets[ind]
            if(self.LabelsCounter[oldLabel]+1 == self.LabelsCounter[oldLabel ^ 1]):
                self.LabelsCounter[oldLabel] += 1
                # self.label_to_indices[oldLabel].remove(ind)
                return self.original_datset[ind]
            if(oldLabel == 1):
                newLabel = 0
            else:
                newLabel = 1
            new_ind = np.random.choice(self.label_to_indices[newLabel])
            self.LabelsCounter[newLabel] += 1
            # self.label_to_indices[newLabel].remove(new_ind)
            return self.original_datset[new_ind]"""

    def __len__(self):
        if self.len_limit:
            return min(len(self.original_datset), self.len_limit)
        return len(self.original_datset)
