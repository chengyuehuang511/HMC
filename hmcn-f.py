import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import random
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Dense(nn.Module):
    """linear + activation"""
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.activation = activation

    def forward(self, x):
        assert x.shape[-1] == self.input_dim
        x = self.linear(x)
        x = self.activation(x)
        return x


class HMCNFModel(nn.Module):
    def __init__(self, features_size, hierarchy, hidden_size=384, beta=0.5, dropout_rate=0.1,
                 if_global=True, h_specific=None):
        """
        feature_size == x.shape[1]
        label_size == sum(hierarchy)
        """
        super().__init__()
        self.features_size = features_size
        self.label_size = sum(hierarchy)
        self.hierarchy = hierarchy
        self.hidden_size = hidden_size
        self.beta = beta
        self.dropout_rate = dropout_rate
        self.if_global = if_global
        self.h_specific = h_specific
        self.softmax = nn.Softmax(dim=-1)

        # if self.h_specific:
        #     for i, h in enumerate(h_specific):
        #         assert self.check_L12_table(h, hierarchy[i], hierarchy[i+1])

        def local_model(input_dim, hidden_dim, num_labels, dropout_rate):
            return nn.Sequential(
                Dense(input_dim, hidden_dim, activation=nn.ReLU()),
                nn.Dropout(dropout_rate),
                Dense(hidden_dim, num_labels, activation=nn.Softmax(dim=-1))
            )

        def global_model(input_dim, output_dim, dropout_rate):
            return nn.Sequential(
                Dense(input_dim, output_dim, activation=nn.ReLU()),
                nn.Dropout(dropout_rate)
            )
        
        self.global_models = nn.ModuleList()
        for i in range(len(hierarchy)):
            if i == 0:
                self.global_models.append(global_model(features_size, hidden_size, dropout_rate))
            else:
                self.global_models.append(global_model(features_size + hidden_size, hidden_size, dropout_rate))
        
        self.local_models = nn.ModuleList([local_model(hidden_size, hidden_size, hierarchy[i], dropout_rate) for i in range(len(hierarchy))])
        
        if if_global:
            self.linear_after_global = nn.Linear(hidden_size, self.label_size)

    def check_L12_table(self, L12_table, L1_labels_num, L2_labels_num):
        """check the Legality of L12_table"""
        if len(L12_table) != L1_labels_num:
            return False
        L2_labels = [num for lst in L12_table for num in lst]
        L2_labels.sort()
        L2_labels_true = [i for i in range(L2_labels_num)]
        if L2_labels != L2_labels_true:
            return False
        return True

    def masking(self, L1, L2, L12_table, L1_labels_num, L2_labels_num, mask_value=-10):
        """
        mask_value must not be too small (e.g. -100), b/c after softmax it will be close to zero,
        then log(->0) will be really small --> CELoss will be very big --> gradient explosion
        """
        assert self.check_L12_table(L12_table, L1_labels_num, L2_labels_num)
        mask = torch.ones_like(L2) * mask_value
        L1_label = L1.argmax(dim=1)
        # Only keep L2_label that have root L1_label
        for i, element in enumerate(L1_label):
            idx = element.item()
            # if parent = -1, then child = -1
            if sum(L1[i]) < 0:
                mask[i, :] = -1
                L2[i, :] += mask[i, :]
            else:
                # if has no child, then child = -1
                if len(L12_table[idx]) == 0:
                    mask[i, :] = -1
                    L2[i, :] += mask[i, :]
                else:
                    mask[i, L12_table[idx]] = 0
                    L2[i, :] += mask[i, :]
                    L2[i, :] = self.softmax(L2[i, :])
        return L2

    def forward(self, x):
        assert x.shape[-1] == self.features_size
        global_models = []
        local_models = []

        for i in range(len(self.hierarchy)):
            if i == 0:
                global_models.append(self.global_models[i](x))
            else:
                global_models.append(self.global_models[i](torch.cat([global_models[i - 1], x], dim=1)))

        for i in range(len(self.hierarchy)):
            local_models.append(self.local_models[i](global_models[i]))

        p_loc = torch.cat(local_models, dim=1)
        labels = p_loc

        # if use global module
        if self.if_global:
            p_glob = self.linear_after_global(global_models[-1])
            cum = 0
            for i in range(len(self.hierarchy)):
                # softmax within hierarchy
                p_glob[:, cum: cum + self.hierarchy[i]] = self.softmax(p_glob[:, cum: cum + self.hierarchy[i]])
                cum += self.hierarchy[i]
            labels = (1-self.beta) * p_glob + self.beta * p_loc

        if self.h_specific:
            assert len(self.h_specific) == len(self.hierarchy) - 1
            cum = 0
            for i, h in enumerate(self.h_specific):
                # assert self.check_L12_table(h, self.hierarchy[i], self.hierarchy[i+1])
                L1 = labels[:, cum: cum + self.hierarchy[i]]
                cum += self.hierarchy[i]
                L2 = labels[:, cum: cum + self.hierarchy[i+1]]
                labels[:, cum: cum + self.hierarchy[i+1]] = self.masking(L1, L2, h, self.hierarchy[i], self.hierarchy[i+1])
        return labels


if __name__=='__main__':

    # x = torch.cat([torch.zeros([1000,77]), torch.ones([628,77])], dim=0)
    # y = torch.cat([torch.zeros([1000,499]), torch.ones([628,499])], dim=0)

    # (1628, 77)
    # (1628, 499)

    torch.manual_seed(1)  # cpu
    torch.cuda.manual_seed(1)  # gpu

    """simulate data"""
    sample_size = 1628
    feature_size = 77
    # hierarchy = [18, 80, 178, 142, 77, 4]

    # hierarchy = [5, 7]
    # h_specific = [[[0], [2, 3, 4], [5], [6], [1]]]

    hierarchy = [2, 2]
    h_specific = [[[], [0, 1]]]

    # hierarchy = [2, 2]
    # h_specific = [[[0], [1]]]

    x = torch.rand(sample_size, feature_size)

    y = []
    for i, h in enumerate(h_specific):
        # -1: doesn't belong to any class within the hierarchy
        if i == 0:
            targets = torch.randint(0, len(h), (sample_size, 1))
            # targets_to_one_hot = F.one_hot(targets, num_classes=hierarchy[i])
            y.append(targets)

        targets = []
        for s in range(sample_size):
            # if parent = -1, then child = -1
            if y[-1][s] == -1:
                targets.append([-1])
            else:
                # if has no child, then child = -1
                if len(h[y[-1][s]]) == 0:
                    targets.append([-1])
                else:
                    targets.append(random.sample(h[y[-1][s]], 1))
        targets = torch.tensor(targets)
        y.append(targets)
    y = torch.cat(y, dim=1)
    """simulation ends"""

    beta = 0.5
    model = HMCNFModel(features_size=feature_size,
                       hierarchy=hierarchy,
                       h_specific=h_specific,
                       hidden_size=384,
                       beta=beta,
                       dropout_rate=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # , weight_decay=1e-5
    criterion = nn.NLLLoss()

    for t in range(200):
        # if t == 3:
        #     print(t)
        y_pred = model(x)
        loss = []
        cum = 0
        for i, h_len in enumerate(hierarchy):
            # delete y==-1 and y_pred <0
            preserve_idx = (y[:, i] != -1) & (y_pred[:, cum: cum + h_len].sum(dim=1) > 0)
            if sum(preserve_idx) > 0:
                loss.append(criterion(torch.log(y_pred[preserve_idx, cum: cum + h_len]), y[preserve_idx, i].long()))
            cum += h_len
        loss = torch.stack(loss).mean()

        y_hat = []
        cum = 0
        for i in range(len(hierarchy)):
            # has the ability to predict
            y_hat_tmp = torch.argmax(y_pred[:, cum: cum + hierarchy[i]], dim=1).reshape(sample_size, 1)
            y_hat_tmp[y_pred[:, cum: cum + h_len].sum(dim=1) < 0] = torch.tensor([-1])
            y_hat.append(y_hat_tmp)
            cum += hierarchy[i]
        y_hat = torch.cat(y_hat, dim=1)

        acc = []
        for i in range(len(hierarchy)):
            # delete y==-1 and y_hat==-1
            preserve_idx = (y[:, i] != -1) & (y_hat[:, i] != -1)
            if sum(preserve_idx) > 0:
                acc.append(accuracy_score(y[preserve_idx, i].numpy(), y_hat[preserve_idx, i].numpy()))
        acc_mean = np.mean(acc)

        print("Epoch {} | Loss: {} | Mean Acc: {}".format(t, loss.item(), acc_mean))

        # with torch.autograd.detect_anomaly():
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # for name, params in model.named_parameters():
        #     print(name)
        #     print(params.requires_grad)
        #     print(params.grad)
