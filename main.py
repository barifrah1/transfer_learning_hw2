import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms


import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import time

import sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc


from net import MyResNet, training_loop, plot_loss_graph, plot_auc_graph
from args import NetArgs
from subloader import SubLoader
from dataloader import CIFAR10_DataLoader_Len_Limited
from torch.utils.data import DataLoader, Dataset

if __name__ == '__main__':

    args = NetArgs()
    """train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]),
        ]
    )"""

    train_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]),
        ]
    )

    """infer_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]),
        ]
    )"""

    infer_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]),
        ]
    )

    labels = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
              'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    print(labels)
    exclude_labels = labels
    del exclude_labels['dog']
    del exclude_labels['cat']
    exclude_labels = list(exclude_labels.values())
    tr_dataset = SubLoader(exclude_labels, "cifar",
                           transform=train_transform, train=True, download=True)
    tr_dataloader = DataLoader(
        CIFAR10_DataLoader_Len_Limited(tr_dataset, int(800*2)), batch_size=args.batch_size
    )
    val_dataset = SubLoader(exclude_labels, "cifar",
                            transform=infer_transform, train=False, download=False)
    val_dataloader = DataLoader(
        CIFAR10_DataLoader_Len_Limited(val_dataset, int(250*2)), batch_size=int(250*2)
    )
    net = MyResNet()
    X_extracted_features_train = torch.empty([0, 512])
    y_extracted_features_train = torch.empty([0]).long()
    X_extracted_features_test = torch.empty([0, 512])
    y_extracted_features_test = torch.empty([0]).long()

    for X, y in tr_dataloader:

        extracted_batch_features = net.extract_features(X)
        X_extracted_features_train = torch.cat(
            [X_extracted_features_train, extracted_batch_features], dim=0)
        y_extracted_features_train = torch.cat(
            [y_extracted_features_train, y], dim=0)

    """for X, y in val_dataloader:
        extracted_batch_features = net.extract_features(X)
        X_extracted_features_test = torch.cat(
            [X_extracted_features_test, extracted_batch_features], dim=0)
        y_extracted_features_test = torch.cat(
            [y_extracted_features_test, y], dim=0)
    print(X_extracted_features_train.shape)
    print(X_extracted_features_test.shape)"""

    tr_loss, val_loss, test_loss, untrained_test_loss = training_loop(
        args,
        net,
        X_extracted_features_train,
        y_extracted_features_train,
        val_dataloader,
        criterion_func=nn.CrossEntropyLoss
    )
    plot_loss_graph(tr_loss, val_loss)
