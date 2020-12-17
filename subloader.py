import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np


class SubLoader(torchvision.datasets.CIFAR10):
    def __init__(self, exclude_list, *args, **kwargs):
        super(SubLoader, self).__init__(*args, **kwargs)

        if exclude_list == []:
            return

        """if self.train:"""
        labels = np.array(self.targets)
        exclude = np.array(exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()
        """else:
            labels = np.array(self.test_labels)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

            self.test_data = self.test_data[mask]
            self.test_labels = labels[mask].tolist()"""
