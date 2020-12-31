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


from net import MyResNet, training_loop, training_loop_with_dataloaders, infer, plot_loss_graph, plot_auc_graph
from args import NetArgs
from subloader import SubLoader
from dataloader import CIFAR10_DataLoader_Len_Limited
from torch.utils.data import DataLoader, Dataset

import os

if __name__ == '__main__':

    args = NetArgs()
    classifyer = nn.Sequential(nn.Linear(512, 2))
    net = MyResNet(classifyer, 18)

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
    RESNET18_EXTRACTED_DATA_EXIST = (os.path.exists(args.X_train_extracted_from_resnet18_file_name) == True and
                                     os.path.exists(args.y_train_extracted_from_resnet18_file_name) == True and
                                     os.path.exists(args.X_test_extracted_from_resnet18_file_name) == True and
                                     os.path.exists(args.y_test_extracted_from_resnet18_file_name) == True)
    RESNET34_EXTRACTED_DATA_EXIST = (os.path.exists(args.X_train_extracted_from_resnet34_file_name) == True and
                                     os.path.exists(args.y_train_extracted_from_resnet34_file_name) == True and
                                     os.path.exists(args.X_test_extracted_from_resnet34_file_name) == True and
                                     os.path.exists(args.y_test_extracted_from_resnet34_file_name) == True)
    FINE_TUNE = False
    LOAD_DATA_FLAG = RESNET18_EXTRACTED_DATA_EXIST and RESNET34_EXTRACTED_DATA_EXIST and FINE_TUNE
    if(LOAD_DATA_FLAG == False):
        labels = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                  'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
        print(labels)
        exclude_labels = labels
        del exclude_labels['dog']
        del exclude_labels['cat']
        exclude_labels = list(exclude_labels.values())

        # Subloader inherits from Dataloader and overrides __init__. It takes only the 'dog' and 'cat' examples from the dataset
        tr_dataset = SubLoader(exclude_labels, "cifar",
                               transform=train_transform, train=True, download=False)
        # CIFAR10_DataLoader_Len_Limited is a class inherits from cifar10Dataloader and limits and number of examples used from the dataset to len_limit
        tr_dataloader = DataLoader(
            CIFAR10_DataLoader_Len_Limited(tr_dataset, int(args.train_size)), batch_size=args.batch_size
        )
        val_dataset = SubLoader(exclude_labels, "cifar",
                                transform=infer_transform, train=False, download=False)
        val_dataloader = DataLoader(
            CIFAR10_DataLoader_Len_Limited(val_dataset, int(args.test_size)), batch_size=int(args.test_size)
        )
    # in case we already extracted the features using resnet18, skip the extraction process
    if(RESNET18_EXTRACTED_DATA_EXIST == False):
        X_extracted_features_train, y_extracted_features_train = net.extract_features_from_dataloader(
            tr_dataloader)
        X_extracted_features_test, y_extracted_features_test = net.extract_features_from_dataloader(
            val_dataloader)
        print(X_extracted_features_train.shape)
        print(X_extracted_features_test.shape)
        # save the extracted features to files
        torch.save(X_extracted_features_train,
                   args.X_train_extracted_from_resnet18_file_name)
        torch.save(y_extracted_features_train,
                   args.y_train_extracted_from_resnet18_file_name)
        torch.save(X_extracted_features_test,
                   args.X_test_extracted_from_resnet18_file_name)
        torch.save(y_extracted_features_test,
                   args.y_test_extracted_from_resnet18_file_name)
    else:
        # load the extracted features files
        X_extracted_features_train = torch.load(
            args.X_train_extracted_from_resnet18_file_name)
        y_extracted_features_train = torch.load(
            args.y_train_extracted_from_resnet18_file_name)
        X_extracted_features_test = torch.load(
            args.X_test_extracted_from_resnet18_file_name)
        y_extracted_features_test = torch.load(
            args.y_test_extracted_from_resnet18_file_name)

    # logistic regression with sklearn

    logisticRegr = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf = logisticRegr.fit(X_extracted_features_train,
                           y_extracted_features_train)
    pred_train = clf.predict(X_extracted_features_train)
    pred_test = clf.predict(X_extracted_features_test)
    # train accuracy and auc
    accuracy_train = clf.score(
        X_extracted_features_train, y_extracted_features_train)
    auc_train = roc_auc_score(pred_train, y_extracted_features_train)
    # test accuracy and auc
    accuracy_test = clf.score(
        X_extracted_features_test, y_extracted_features_test)
    auc_test = roc_auc_score(pred_test, y_extracted_features_test)
    print("train results:")
    print('accuracy:', accuracy_train)
    print('auc', auc_train)
    print("test results:")
    print('accuracy:', accuracy_test)
    print('auc', auc_test)

    # logistic regression with pytorch and resnet18
    tr_loss, val_loss, test_loss, tr_auc, val_auc, untrained_test_loss, untrained_test_auc = training_loop(
        args,
        net,
        X_extracted_features_train,
        y_extracted_features_train,
        X_extracted_features_test,
        y_extracted_features_test,
        val_dataloader=None,
        criterion_func=nn.CrossEntropyLoss
    )
    plot_loss_graph(tr_loss, val_loss)
    plot_auc_graph(tr_auc, val_auc)

    """
    # models with pytorch and resnet18 with 3 fc layers in the head
    classifyer_with_3_layers = nn.Sequential(
        nn.Linear(512, 20), nn.Linear(20, 10), nn.Linear(10, 2))
    net_with_3_layers_in_head = MyResNet(classifyer_with_3_layers, 18)
    tr_loss_3_layers, val_loss_3_layers, test_loss_3_layers, tr_auc_3_layers, val_auc_3_layers, untrained_test_loss_3_layers, untrained_test_auc_3_layers = training_loop(
        args,
        net_with_3_layers_in_head,
        X_extracted_features_train,
        y_extracted_features_train,
        X_extracted_features_test,
        y_extracted_features_test,
        val_dataloader=None,
        criterion_func=nn.CrossEntropyLoss
    )
    plot_loss_graph(tr_loss_3_layers, val_loss_3_layers)
    plot_auc_graph(tr_auc_3_layers, val_auc_3_layers)

    # logistic regression with resnet34 and pytorch
    classifyer34 = nn.Sequential(nn.Linear(512, 2))
    net34 = MyResNet(classifyer34, 34)
    # in case we already extracted the features using resnet34, skip the extraction process
    if(RESNET34_EXTRACTED_DATA_EXIST == False):

        X_extracted_features_train34, y_extracted_features_train34 = net34.extract_features_from_dataloader(
            tr_dataloader)
        X_extracted_features_test34, y_extracted_features_test34 = net34.extract_features_from_dataloader(
            val_dataloader)
        print(X_extracted_features_train34.shape)
        print(X_extracted_features_test34.shape)
        torch.save(X_extracted_features_train,
                   args.X_train_extracted_from_resnet34_file_name)
        torch.save(y_extracted_features_train,
                   args.y_train_extracted_from_resnet34_file_name)
        torch.save(X_extracted_features_test,
                   args.X_test_extracted_from_resnet34_file_name)
        torch.save(y_extracted_features_test,
                   args.y_test_extracted_from_resnet34_file_name)
    else:
        # load the files
        X_extracted_features_train34 = torch.load(
            args.X_train_extracted_from_resnet34_file_name)
        y_extracted_features_train34 = torch.load(
            args.y_train_extracted_from_resnet34_file_name)
        X_extracted_features_test34 = torch.load(
            args.X_test_extracted_from_resnet34_file_name)
        y_extracted_features_test34 = torch.load(
            args.y_test_extracted_from_resnet34_file_name)
    tr_loss34, val_loss34, test_loss34, tr_auc34, val_auc34, untrained_test_loss34, untrained_test_auc34 = training_loop(
        args,
        net34,
        X_extracted_features_train34,
        y_extracted_features_train34,
        X_extracted_features_test34,
        y_extracted_features_test34,
        val_dataloader=None,
        criterion_func=nn.CrossEntropyLoss
    )
    plot_loss_graph(tr_loss34, val_loss34)
    plot_auc_graph(tr_auc34, val_auc34)

    # models with pytorch and resnet34 with 3 fc layers in the head
    classifyer34_with_3_layers = nn.Sequential(
        nn.Linear(512, 20), nn.Linear(20, 10), nn.Linear(10, 2))
    net34_with_3_layers_in_head = MyResNet(classifyer34_with_3_layers, 34)
    tr_loss34_3_layers_in_head, val_loss34_3_layers_in_head, test_loss34_3_layers_in_head, tr_auc34_3_layers_in_head, val_auc34_3_layers_in_head, untrained_test_loss34_3_layers_in_head, untrained_test_auc34_3_layers_in_head = training_loop(
        args,
        net34_with_3_layers_in_head,
        X_extracted_features_train34,
        y_extracted_features_train34,
        X_extracted_features_test34,
        y_extracted_features_test34,
        val_dataloader=None,
        criterion_func=nn.CrossEntropyLoss
    )
    plot_loss_graph(tr_loss34, val_loss34)
    plot_auc_graph(tr_auc34, val_auc34)"""

    # finetuning
# net_param=net.features
print('finetuning')
i = 0
for param in net.features.parameters():
    param.requires_grad = True
    #i += 1


"""ct = 0
for child in net.children():
    ct += 1
    if ct < 7:
        for param in child.parameters():
            param.requires_grad = True
"""
args.lr = 1e-6
tr_loss_ft, val_loss_ft, test_loss_ft, tr_auc_ft, val_auc_ft, untrained_test_loss_ft, untrained_test_auc_ft = training_loop_with_dataloaders(
    args,
    net,
    tr_dataloader=tr_dataloader,
    val_dataloader=val_dataloader,
    criterion_func=nn.CrossEntropyLoss
)
plot_loss_graph(tr_loss_ft, val_loss_ft)
plot_auc_graph(tr_auc_ft, val_auc_ft)
