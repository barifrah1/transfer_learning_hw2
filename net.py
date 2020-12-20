import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt


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
        self.classifyer = nn.Sequential(
            nn.Linear(num_ftrs, int(num_ftrs/2)), nn.ReLU(), nn.Linear(int(num_ftrs/2), int(num_ftrs/4)), nn.ReLU(), nn.Linear(int(num_ftrs/4), 2))

    def extract_features(self, X):
        self.features.eval()
        with torch.no_grad():
            return self.features(X)

    def forward(self, X):
        return self.classifyer(X)


def infer(net, X, y, criterion):
    net.eval()
    running_loss = 0
    with torch.no_grad():
        pred = net(X)
        loss = criterion(pred, y).item()
    return loss


def training_loop(
    args,
    net,
    X_train,
    y_train,
    X_val,
    y_val,
    criterion_func=nn.CrossEntropyLoss,
    optimizer_func=optim.SGD,
):
    criterion = criterion_func()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    tr_loss, val_loss = [None] * args.num_epochs, [None] * args.num_epochs
    test_loss, untrained_test_loss = None, None
    # Note that I moved the inferences to a function because it was too much code duplication to read.
    # calculate error before training
    untrained_test_loss = infer(net, X_val, y_val, criterion)
    for epoch in range(args.num_epochs):
        net.train()
        running_tr_loss = 0
        data_size = len(X_train)
        no_of_batches = data_size // args.batch_size
        for i in tqdm(range(no_of_batches)):
            start = i*args.batch_size
            end = i*args.batch_size + args.batch_size
            x = X_train[start:end]
            y = y_train[start:end]
            optimizer.zero_grad()
            pred = net(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_tr_loss += loss*args.batch_size
        tr_loss[epoch] = running_tr_loss.item() / data_size

        val_loss[epoch] = infer(net, X_val, y_val, criterion)
        print(
            f"Train loss: {tr_loss[epoch]:.2e}, Val loss: {val_loss[epoch]:.2e}")
        if epoch >= args.early_stopping_num_epochs:
            improvement = (
                val_loss[epoch - args.early_stopping_num_epochs] -
                val_loss[epoch]
            )
            if improvement < args.early_stopping_min_improvement:
                break
        # torch.save(net.state_dict(), f"my_net_{epoch}.pt")

    test_loss = infer(net, X_val, y_val, criterion)
    print(f"Stopped training after {epoch+1}/{args.num_epochs} epochs.")
    print(
        f"The loss is {untrained_test_loss:.2e} before training and {test_loss:.2e} after training."
    )
    print(
        f"The training and validation losses are "
        f"\n\t{tr_loss}, \n\t{val_loss}, \n\tover the training epochs, respectively."
    )
    # torch.save(
    #     {
    #         "epoch": epoch,
    #         "model_state_dict": net.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "train_loss": tr_loss,
    #         "validation_loss": val_loss,
    #         "untrained_loss": untrained_test_loss,
    #     },
    #     "net_training_state.pt",
    # )
    return tr_loss, val_loss, test_loss, untrained_test_loss


def plot_loss_graph(loss_list, train_or_val: str):
    loss_values = loss_list
    epochs = range(1, len(loss_values)+1)
    plt.plot(epochs, loss_values, label=f'{train_or_val} Loss by epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_auc_graph(auc, train_or_val: str):
    epochs = range(1, len(auc)+1)
    plt.plot(epochs, auc, label=f'{train_or_val} auc by epoch')
    plt.xlabel('Epochs')
    plt.ylabel('auc')
    plt.legend()

    plt.show()


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
    plt.show()
