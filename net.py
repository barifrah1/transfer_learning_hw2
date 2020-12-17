import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms


class MyResNet(nn.Module):

    def __init__(self):
        super(MyResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.features = resnet
        # Freezing resnet params:
        for param in self.features.parameters():
            param.requires_grad = False
        # fc layer with softMax for classifying
        self.classifyer = nn.Sequential(nn.Linear(num_ftrs, 2), nn.Softmax())

    def extract_features(self, X):
        self.features.eval()
        with torch.no_grad():
            return self.features(X)
