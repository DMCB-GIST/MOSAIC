import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

loss_func = nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean')
loss_func2 = nn.MarginRankingLoss()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model = model
            self.val_loss_min = val_loss
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_model = model
            self.counter = 0



class CNN_1(nn.Module):
    def __init__(self, dim, int_dim, shell):
        super(CNN_1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=shell, stride=shell, padding=0),
            nn.BatchNorm1d(16)
        )
        self.fc = nn.Linear(dim, int_dim)
        self.fc2 = nn.Linear(int_dim, 1)

        self.act = nn.Tanh()
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x):

        x = x.view(x.shape[0], 1, -1)
        out = self.conv(x)
        out = self.act(out)
        out = self.pool(out)
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        feat = self.fc(out)
        logit = self.fc2(feat)
        logit = logit.view(-1, )

        return logit, feat


class CNN_2(nn.Module):
    def __init__(self, dim, int_dim, shell):
        super(CNN_2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=shell, stride=shell, padding=0),
            nn.BatchNorm1d(16)
        )
        self.fc = nn.Linear(dim, int_dim)
        self.fc2 = nn.Linear(int_dim, 1)

        self.act = nn.Tanh()
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        out = self.conv(x)
        out = self.act(out)
        out = self.pool(out)
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        feat = self.fc(out)
        logit = self.fc2(feat)
        logit = logit.view(-1, )

        return logit, feat

def train_model(train_x, valid_x, tr_y, val_y, model, patience, n_epochs, optimizer, rg):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model.train()

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        optimizer.zero_grad()

        output, feat = model(train_x)
        # calculate the loss
        alpha_t = 0.8 * ((early_stopping.counter + 1) / patience)
        alpha_t = max(0, alpha_t)
        if epoch == 1:
            before_predict = tr_y.float()

        loss1 = loss_func(output, tr_y.float())
        target = Variable(torch.where(tr_y.float() > 0, 1, -1))
        loss2 = loss_func2(torch.sigmoid(output), before_predict, target)
        loss = ((1 - alpha_t) * loss1 + (alpha_t) * loss2)

        norm = torch.cuda.FloatTensor([0])
        for parameter in model.parameters():
            norm += torch.norm(parameter, p=1)

        loss = loss + norm * rg

        before_predict = torch.sigmoid(output).detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            model.eval()
            output, feat = model(valid_x)
            loss = loss_func(output, val_y.float())

            # record validation loss
            valid_losses.append(loss.item())

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        #print(print_msg)

        train_losses = []
        valid_losses = []
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            break

    return early_stopping.best_model, early_stopping.val_loss_min
