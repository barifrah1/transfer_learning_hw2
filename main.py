import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import time

import sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc


from net import MyResNet
from args import NetArgs
from subloader import SubLoader
from dataloader import CIFAR10_DataLoader_Len_Limited
from torch.utils.data import DataLoader, Dataset

if __name__ == '__main__':

    args = NetArgs()
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]),
        ]
    )
    infer_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]),
        ]
    )

    tr_dataset = datasets.CIFAR10(
        "cifar", transform=train_transform, train=True, download=True)
    labels = tr_dataset.class_to_idx
    exclude_labels = labels
    del exclude_labels['dog']
    del exclude_labels['cat']
    exclude_labels = list(exclude_labels.values())
    #include_labels = [labels["cat"], labels["dog"]]
    tr_dataset = SubLoader(exclude_labels, "cifar",
                           transform=train_transform, train=True, download=True)
    tr_dataloader = DataLoader(
        CIFAR10_DataLoader_Len_Limited(tr_dataset, int(1e2)), batch_size=args.batch_size
    )
    val_dataset = SubLoader(exclude_labels, "cifar",
                            transform=train_transform, train=False, download=True)
    val_dataloader = DataLoader(
        CIFAR10_DataLoader_Len_Limited(val_dataset, int(1e1)), batch_size=args.batch_size
    )
    res = MyResNet()
    X_extracted_features = torch.empty([0, 512])
    y_extracted_features = torch.empty([0]).long()
    for X, y in tr_dataloader:
        extracted_batch_features = res.extract_features(X)
        X_extracted_features = torch.cat(
            [X_extracted_features, extracted_batch_features], dim=0)
        y_extracted_features = torch.cat([y_extracted_features, y], dim=0)
    print(X_extracted_features.shape)
