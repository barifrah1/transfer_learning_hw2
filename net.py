import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class MyResNet(nn.Module):

    def __init__(self, classifyer, res34or18):
        super(MyResNet, self).__init__()
        if(res34or18 == 18):
            resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet34(pretrained=True)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.features = resnet
        # Freezing resnet params:
        for param in self.features.parameters():
            param.requires_grad = False
        # fc layer with softMax for classifying
        self.classifyer = classifyer

    def __extract_features__(self, X):
        self.features.eval()
        with torch.no_grad():
            return self.features(X)

    def extract_features_from_dataloader(self, dataloader):
        X_extracted_features = torch.empty([0, 512])
        y_extracted_features = torch.empty([0]).long()
        for X, y in dataloader:
            extracted_batch_features = self.__extract_features__(X)
            X_extracted_features = torch.cat(
                [X_extracted_features, extracted_batch_features], dim=0)
            y_extracted_features = torch.cat(
                [y_extracted_features, y], dim=0)

        return X_extracted_features, y_extracted_features

    def forward(self, X):
        if(len(X.shape) != 4):
            return self.classifyer(X)
        else:
            X = self.features(X)
            return self.classifyer(X)


# can pass to function dataloader or X,y
def infer(net, criterion, X=None, y=None, dataloader=None):
    net.eval()
    running_loss = 0
    running_auc = 0
    num_of_rows = 0
    with torch.no_grad():
        if(dataloader == None):
            pred = net(X)
            loss = criterion(pred, y).item()
            auc = roc_auc_score(y, pred.numpy()[:, 1])
            return loss, auc
        else:
            for X_batch, y_batch in dataloader:
                pred = net(X_batch)
                loss = criterion(pred, y_batch).item()
                auc = roc_auc_score(y_batch, pred.numpy()[:, 1])
                running_loss += loss*y_batch.shape[0]
                running_auc += auc*y_batch.shape[0]
                num_of_rows += y_batch.shape[0]
            return running_loss / num_of_rows, running_auc / num_of_rows


def training_loop(
    args,
    net,
    X_train,
    y_train,
    X_test,
    y_test,
    tr_dataloader=None,
    val_dataloader=None,
    criterion_func=nn.CrossEntropyLoss,
    optimizer_func=optim.SGD,
):
    criterion = criterion_func()
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    tr_loss, val_loss = [None] * args.num_epochs, [None] * args.num_epochs
    tr_auc, val_auc = [None] * args.num_epochs, [None] * args.num_epochs
    test_loss, untrained_test_loss = None, None
    test_auc, untrained_test_auc = None, None
    # Note that I moved the inferences to a function because it was too much code duplication to read.
    # calculate error before training
    auc_and_loss = infer(
        net,  criterion, X=X_test, y=y_test, dataloader=val_dataloader)
    untrained_test_loss = auc_and_loss[0]
    untrained_test_auc = auc_and_loss[1]
    for epoch in range(args.num_epochs):
        net.train()
        running_tr_loss = 0
        running_tr_auc = 0
        data_size = len(X_train)
        if(data_size % args.batch_size == 0):
            no_of_batches = data_size // args.batch_size
        else:
            no_of_batches = (data_size // args.batch_size)+1
        for i in tqdm(range(no_of_batches)):
            start = i*args.batch_size
            end = i*args.batch_size + args.batch_size
            x = X_train[start:end]
            y = y_train[start:end]
            optimizer.zero_grad()
            pred = net(x)
            auc = roc_auc_score(y, pred.detach().numpy()[:, 1])
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_tr_loss += loss*x.shape[0]
            running_tr_auc += auc*x.shape[0]

        tr_loss[epoch] = running_tr_loss.item() / data_size
        tr_auc[epoch] = running_tr_auc.item() / data_size
        auc_and_loss = infer(
            net, criterion, X=X_test, y=y_test, dataloader=None)
        val_loss[epoch] = auc_and_loss[0]
        val_auc[epoch] = auc_and_loss[1]
        print(
            f"Train loss: {tr_loss[epoch]:.2e}, Val loss: {val_loss[epoch]:.2e}")
        if epoch >= args.early_stopping_num_epochs:
            improvement = (
                val_loss[epoch - args.early_stopping_num_epochs] -
                val_loss[epoch]
            )
            if improvement < args.early_stopping_min_improvement:
                break
    auc_and_loss = infer(net, criterion, X=X_test, y=y_test, dataloader=None)
    test_loss = auc_and_loss[0]
    test_auc = auc_and_loss[1]
    print(f"Stopped training after {epoch+1}/{args.num_epochs} epochs.")
    print(
        f"The loss is {untrained_test_loss:.2e} before training and {test_loss:.2e} after training."
    )
    print(
        f"The training and validation losses are "
        f"\n\t{tr_loss}, \n\t{val_loss}, \n\tover the training epochs, respectively."
    )
    return tr_loss, val_loss, test_loss, tr_auc, val_auc, untrained_test_loss, untrained_test_auc


def training_loop_with_dataloaders(
    args,
    net,
    tr_dataloader=None,
    val_dataloader=None,
    criterion_func=nn.CrossEntropyLoss,
    optimizer_func=optim.SGD,
):
    criterion = criterion_func()
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    tr_loss, val_loss = [None] * args.num_epochs, [None] * args.num_epochs
    tr_auc, val_auc = [None] * args.num_epochs, [None] * args.num_epochs
    test_loss, untrained_test_loss = None, None
    test_auc, untrained_test_auc = None, None
    # Note that I moved the inferences to a function because it was too much code duplication to read.
    # calculate error before training
    auc_and_loss = infer(
        net,  criterion, X=None, y=None, dataloader=val_dataloader)
    untrained_test_loss = auc_and_loss[0]
    untrained_test_auc = auc_and_loss[1]
    for epoch in range(args.num_epochs):
        net.train()
        running_tr_loss = 0
        running_tr_auc = 0
        for x, y in tr_dataloader:
            print(1)
            optimizer.zero_grad()
            pred = net(x)
            auc = roc_auc_score(y, pred.detach().numpy()[:, 1])
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_tr_loss += loss*x.shape[0]
            running_tr_auc += auc*x.shape[0]

        tr_loss[epoch] = running_tr_loss.item() / args.train_size
        tr_auc[epoch] = running_tr_auc.item() / args.train_size
        auc_and_loss = infer(
            net, criterion, X=None, y=None, dataloader=val_dataloader)
        val_loss[epoch] = auc_and_loss[0]
        val_auc[epoch] = auc_and_loss[1]
        print(
            f"Train loss: {tr_loss[epoch]:.2e}, Val loss: {val_loss[epoch]:.2e}")
        if epoch >= args.early_stopping_num_epochs:
            improvement = (
                val_loss[epoch - args.early_stopping_num_epochs] -
                val_loss[epoch]
            )
            if improvement < args.early_stopping_min_improvement:
                break
    auc_and_loss = infer(net, criterion, X=None, y=None,
                         dataloader=val_dataloader)
    test_loss = auc_and_loss[0]
    test_auc = auc_and_loss[1]
    print(f"Stopped training after {epoch+1}/{args.num_epochs} epochs.")
    print(
        f"The loss is {untrained_test_loss:.2e} before training and {test_loss:.2e} after training."
    )
    print(
        f"The training and validation losses are "
        f"\n\t{tr_loss}, \n\t{val_loss}, \n\tover the training epochs, respectively."
    )
    return tr_loss, val_loss, test_loss, tr_auc, val_auc, untrained_test_loss, untrained_test_auc


def plot_loss_graph(train_loss_list, validation_loss_list):
    plt.plot(train_loss_list, 'g', label='Training loss')
    plt.plot(validation_loss_list, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=False)


def plot_auc_graph(auc_train_list, auc_val_list):
    plt.plot(auc_train_list, 'g', label='Training AUC')
    plt.plot(auc_val_list, 'b', label='validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.show(block=False)


def plot_roc(fpr, tpr):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve on test set')
    plt.legend(loc="lower right")
    plt.show(block=False)
