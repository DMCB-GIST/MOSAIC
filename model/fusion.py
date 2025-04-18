import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from embedding import EarlyStopping
import torch.nn.functional as F
import random
import copy
loss_func = nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean')

class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives):
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

class GCN(nn.Module):
    def __init__(self, num_feature, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_feature, num_feature, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x)
        h = h - x
        h = self.relu(self.conv2(h.permute(0, 2, 1)).permute(0, 2, 1))
        return h

class BilateralGCN(nn.Module):
    def __init__(self, in_feature, num_node):
        super().__init__()
        self.gcn = GCN(in_feature, num_node)

    def forward(self, x, y):
        fusion = x + y
        fusion = self.gcn(fusion)
        x = x + fusion
        y = y + fusion

        return x, y

class clffusion(nn.Module):
    def __init__(self, in_dim):
        super(clffusion, self).__init__()
        self.num_n = in_dim // 4
        self.num_s = in_dim // 2

        self.conv1 = nn.Conv1d(in_dim, self.num_n, kernel_size=1)
        self.conv2 = nn.Conv1d(in_dim, self.num_s, kernel_size=1)

        self.BGCN = BilateralGCN(self.num_s, self.num_n)

        self.conv = nn.Conv1d(self.num_s, in_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_dim)
        self.in_dim = in_dim

        self.clf1 = nn.Linear(in_dim, 1)
        self.clf2 = nn.Linear(in_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        n = x1.size(0)
        e = torch.bmm(x1.reshape(n, 1, -1).permute(0, 2, 1), x2.reshape(n, 1, -1))
        s1 = self.softmax(e.permute(0, 2, 1)).permute(0, 2, 1)
        s2 = self.softmax(e).permute(0, 2, 1)

        x1 = torch.bmm(x1.reshape(n, 1, -1), s1)
        x2 = torch.bmm(x2.reshape(n, 1, -1), s2)

        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        x1_v = self.conv1(x1)
        x2_v = self.conv1(x2)
        x1_w = self.conv2(x1)
        x2_w = self.conv2(x2)

        x1_g = torch.bmm(x1_v, torch.transpose(x1_w, 1, 2))
        x2_g = torch.bmm(x2_v, torch.transpose(x2_w, 1, 2))

        x1_g, x2_g = self.BGCN(x1_g, x2_g)
        x1_g = torch.bmm(torch.transpose(x1_v, 1, 2), x1_g).view(n, self.num_s, -1)
        x2_g = torch.bmm(torch.transpose(x2_v, 1, 2), x2_g).view(n, self.num_s, -1)

        x1_fusion = self.conv(x1_g)
        x2_fusion = self.conv(x2_g)

        x1_fusion = F.relu(self.bn(x1_fusion))
        x2_fusion = F.relu(self.bn(x2_fusion))

        res1 = (x1 + x1_fusion).view(n, self.in_dim)
        res2 = (x2 + x2_fusion).view(n, self.in_dim)

        logit1 = self.clf1(res1).view(-1)
        logit2 = self.clf2(res2).view(-1)

        return logit1, logit2


def train_model(tr1, tr2, val1, val2, clf, patience, n_epochs, tr_y_clf, val_y, hard_ind, opt):
    train_losses = []
    valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    clf.train()
    for epoch in range(1, n_epochs + 1):

        if early_stopping.counter < (patience//2):
            tr_x1 = tr1[~hard_ind]
            tr_x2 = tr2[~hard_ind]
            tr_y = tr_y_clf[~hard_ind]
        else:
            tr_x1 = tr1
            tr_x2 = tr2
            tr_y = tr_y_clf

        ###################
        # train the model #
        ###################
        opt.zero_grad()
        output1, output2 = clf(tr_x1, tr_x2)

        loss1 = loss_func(output1, tr_y.float())
        loss2 = loss_func(output2, tr_y.float())
        loss = loss1 + loss2
        losses = [loss1, loss2]

        opt.zero_grad()
        opt.pc_backward(losses)
        opt.step()
        train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            clf.eval()
            output1, output2 = clf(val1, val2)

            loss1 = loss_func(output1, val_y.float())
            loss2 = loss_func(output2, val_y.float())
            loss = loss1 + loss2

            valid_losses.append(loss.item())
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, clf)

        if early_stopping.early_stop:
            break
    best_clf = early_stopping.best_model

    return best_clf, early_stopping.val_loss_min