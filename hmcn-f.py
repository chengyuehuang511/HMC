import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
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
    def __init__(self, features_size, hierarchy, hidden_size=384, beta=0.5, dropout_rate=0.1, if_global=True, h_specific=None):
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
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        assert x.shape[-1] == self.features_size
        global_models = []
        local_models = []

        for i in range(len(hierarchy)):
            if i == 0:
                global_models.append(self.global_models[i](x))
            else:
                global_models.append(self.global_models[i](torch.cat([global_models[i - 1], x], dim=1)))

        for i in range(len(hierarchy)):
            local_models.append(self.local_models[i](global_models[i]))

        p_loc = torch.cat(local_models, dim=1)

        # if use global module
        if self.if_global:
            p_glob = self.linear_after_global(global_models[-1])
            cum = 0
            for i in range(len(hierarchy)):
                # softmax within hierarchy
                p_glob[:, cum: cum + hierarchy[i]] = self.softmax(p_glob[:, cum: cum + hierarchy[i]])
                cum += hierarchy[i]
            labels = (1-self.beta) * p_glob + self.beta * p_loc
            return labels
        return p_loc


if __name__=='__main__':

    # x = torch.cat([torch.zeros([1000,77]), torch.ones([628,77])], dim=0)
    # y = torch.cat([torch.zeros([1000,499]), torch.ones([628,499])], dim=0)

    # (1628, 77)
    # (1628, 499)

    
    torch.manual_seed(1)  # cpu
    torch.cuda.manual_seed(1)  # gpu

    sample_size = 1628
    feature_size = 77
    hierarchy = [18, 80, 178, 142, 77, 4]

    """
    hierarchy = [5, 7, 4]
    level_1_2 = [[0], [2, 3, 4], [5], [6], [1]]
    """

    x = torch.rand(sample_size, feature_size)

    y = []
    for i in range(len(hierarchy)):
        # -1: doesn't belong to any class within the hierarchy
        targets = torch.randint(-1, hierarchy[i], (sample_size, 1))
        # targets_to_one_hot = F.one_hot(targets, num_classes=hierarchy[i])
        y.append(targets)
    y = torch.cat(y, dim=1)

    beta = 0.5
    model = HMCNFModel(features_size=feature_size, 
                       hierarchy=hierarchy, 
                       hidden_size=384, 
                       beta=0.5, 
                       dropout_rate=0.1)
    y_pred = model(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.NLLLoss()
    
    for t in range(2):
        y_pred = model(x)
        loss = []
        cum = 0
        for i, h_len in enumerate(hierarchy):
            # delete y==-1
            loss.append(criterion(torch.log(y_pred[y[:, i]!=-1, cum: cum + h_len]), y[y[:, i]!=-1, i].long()))
            cum += h_len
        loss = torch.stack(loss).mean()
        
        y_hat = []
        cum = 0
        for i in range(len(hierarchy)):
            # has the abbility to predict
            y_hat.append(torch.argmax(y_pred[:, cum: cum + hierarchy[i]], dim=1).reshape(sample_size, 1))
            cum += hierarchy[i]
        y_hat = torch.cat(y_hat, dim=1)
        
        acc = []
        for i in range(len(hierarchy)):
            # delete y==-1
            acc.append(accuracy_score(y[y[:, i]!=-1, i].numpy(), y_hat[y[:, i]!=-1, i].numpy()))
        acc_mean = np.mean(acc)
        
        print("Epoch {} | Loss: {} | Mean Acc: {}".format(t, loss.item(), acc_mean))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()