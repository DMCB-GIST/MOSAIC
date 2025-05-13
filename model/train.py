import embedding
import fusion
from embedding import EarlyStopping
from fusion import PCGrad
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import tqdm
import gc
import sklearn.metrics as metrics

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run prediction pipeline with test dataset.")
    parser.add_argument('--data_dir', type=str, required=True, default= './data',
                        help='Path to the input data directory (e.g., ./data)')
    return parser.parse_args()

args = parse_args()
type = args.data_dir

rg = 0.001
learning_rate = 0.0001
lr1 = 0.0001
lr2 = 0.0001
n_epochs = 3000
patience = 30

x1 = pd.read_csv(type + 'mRNA.csv', index_col=0, delimiter=',')
x2 = pd.read_csv(type + 'miRNA.csv', index_col=0, delimiter=',')
label = pd.read_csv(type + 'label.csv',  index_col=0,delimiter=',')
print(x1.shape)
print(x2.shape)
x1 = x1.fillna(0)
x2 = x2.fillna(0)
labels = np.array(label).flatten().astype(np.float32)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed_everything(111)

shell_list = list(range(10, 300, 30))
int_dim_list = [1000, 500, 100, 50]
rg_list = [0.0, 0.001,0.0001]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

n_f1 = x1.shape[0]
n_f2 = x2.shape[0]

x1 = torch.tensor(x1.values)
x2 = torch.tensor(x2.values)

t1_res = []
t2_res = []

for it in range(0, 10):

    t1_auc = []
    t2_auc = []

    kf = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=it)
    train_index = []
    valid_index = []
    test_index = []
    for tr_index, te_index in kf.split(np.transpose(x1), labels):
        random.shuffle(tr_index)
        tmp = int(0.2 * len(tr_index))

        train_index.append(tr_index[tmp + 1:])
        valid_index.append(tr_index[0:tmp])
        test_index.append(te_index)

    seed_everything(it)

    for fold in range(1, 6):
        print("------------- Fold : " + str(fold) + "-------------")
        y = torch.tensor(labels)
        train_y = y[train_index[fold - 1]]
        valid_y = y[valid_index[fold - 1]]
        test_y = y[test_index[fold - 1]]

        train_y = Variable(train_y.type(torch.LongTensor)).to(device)
        valid_y = Variable(valid_y.type(torch.LongTensor)).to(device)
        test_y = Variable(test_y.type(torch.LongTensor))

        tr_x1 = x1[:, train_index[fold - 1]].t()
        val_x1 = x1[:, valid_index[fold - 1]].t()
        te_x1 = x1[:, test_index[fold - 1]].t()
        tr_x1 = torch.reshape(tr_x1, (tr_x1.shape[0], tr_x1.shape[1], 1))
        val_x1 = torch.reshape(val_x1, (val_x1.shape[0], val_x1.shape[1], 1))
        te_x1 = torch.reshape(te_x1, (te_x1.shape[0], te_x1.shape[1], 1))

        tr_x1 = Variable(tr_x1.type(torch.FloatTensor)).to(device)
        val_x1 = Variable(val_x1.type(torch.FloatTensor)).to(device)
        te_x1 = Variable(te_x1.type(torch.FloatTensor)).to(device)

        tr_x2 = x2[:, train_index[fold - 1]].t()
        val_x2 = x2[:, valid_index[fold - 1]].t()
        te_x2 = x2[:, test_index[fold - 1]].t()

        tr_x2 = torch.reshape(tr_x2, (tr_x2.shape[0], tr_x2.shape[1], 1))
        val_x2 = torch.reshape(val_x2, (val_x2.shape[0], val_x2.shape[1], 1))
        te_x2 = torch.reshape(te_x2, (te_x2.shape[0], te_x2.shape[1], 1))

        tr_x2 = Variable(tr_x2.type(torch.FloatTensor)).to(device)
        val_x2 = Variable(val_x2.type(torch.FloatTensor)).to(device)
        te_x2 = Variable(te_x2.type(torch.FloatTensor)).to(device)

        best_val = 9999
        for in_dim in tqdm.tqdm(range(0, len(int_dim_list))):
            int_dim = int_dim_list[in_dim]

            # Omics1
            best_val1 = 9999
            for s in range(0, len(shell_list)):
                for r in range(0, len(rg_list)):
                    rg = rg_list[r]
                    shell = shell_list[s]
                    dim = 16 * int(int(n_f1 / shell) / 2)
                    model = embedding.CNN_1(dim, int_dim, shell).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr1)
                    m1, valid_loss = embedding.train_model(tr_x1, val_x1, train_y, valid_y, model, patience, 1000,
                                                           optimizer, rg)
                    if valid_loss < best_val1:
                        best_val1 = valid_loss
                        best_model1 = m1
                        shell1 = shell
            # Embedding
            logit_x1, tr_f1 = best_model1(tr_x1)
            _, val_f1 = best_model1(val_x1)
            _, te_f1 = best_model1(te_x1)

            # Omics2
            best_val2 = 9999
            for s in range(0, len(shell_list)):
                for r in range(0, len(rg_list)):
                    rg = rg_list[r]
                    shell = shell_list[s]
                    dim = 16 * int(int(n_f2 / shell) / 2)
                    model = embedding.CNN_2(dim, int_dim, shell).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr2)

                    m2, valid_loss = embedding.train_model(tr_x2, val_x2, train_y, valid_y, model, patience, 1000,
                                                           optimizer, rg)
                    if valid_loss < best_val2:
                        best_val2 = valid_loss
                        best_model2 = m2
                        shell2 = shell
            # Embedding
            logit_x2, tr_f2 = best_model2(tr_x2)
            _, val_f2 = best_model2(val_x2)
            _, te_f2 = best_model2(te_x2)

            # Delete for memory
            del (m1)
            del (m2)
            del (best_model1)
            del (best_model2)
            gc.collect()

            res1 = abs(train_y - torch.sigmoid(logit_x1))
            logit_mean1 = torch.mean(res1)
            ind1 = (res1 > logit_mean1)
            res2 = abs(train_y - torch.sigmoid(logit_x2))
            logit_mean2 = torch.mean(res2)
            ind2 = (res2 > logit_mean2)

            # Training fusion block
            inter_ind = (ind1 & ind2)
            clf = fusion.clffusion(int_dim).cuda()
            optimizer_f = PCGrad(torch.optim.SGD(clf.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005,
                                                 nesterov=True))
            best, f_val_loss = fusion.train_model(tr_f1.detach(), tr_f2.detach(), val_f1.detach(), val_f2.detach(),
                                                  clf, patience, 300, train_y, valid_y, inter_ind, optimizer_f)
            if f_val_loss < best_val:
                best_val = f_val_loss
                res1, res2 = best(te_f1.detach(), te_f2.detach())
                y_pred1 = torch.sigmoid(res1).detach().cpu()
                y_pred2 = torch.sigmoid(res2).detach().cpu()

                # Task1 performance
                fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred1)
                roc_auc1 = metrics.auc(fpr, tpr)

                # Task2 performance
                fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred2)
                roc_auc2 = metrics.auc(fpr, tpr)

        t1_auc.append(roc_auc1)
        print("Task1 test AUC:", roc_auc1)
        t2_auc.append(roc_auc2)
        print("Task2 test AUC:", roc_auc2)

    t1_res.append(sum(t1_auc) / len(t1_auc))
    t2_res.append(sum(t2_auc) / len(t2_auc))

    # Average performance at each epoch
    print("Task1: AUC", sum(t1_auc) / len(t1_auc))
    print("Task2: AUC", sum(t2_auc) / len(t2_auc))

# Final performance
print("-----------------------------------")
print("Task1 Average: AUC", sum(t1_res) / len(t1_res), np.std(t1_res))
print("Task2 Average: AUC", sum(t2_res) / len(t2_res), np.std(t2_res))
