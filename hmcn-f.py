import torch
import torch.nn as nn
import numpy as np


class Dense(nn.Module):
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
    def __init__(self, features_size, label_size, hierarchy, hidden_size=384, beta=0.5, dropout_rate=0.1):
        """
        feature_size == x.shape[1]
        label_size == sum(hierarchy)
        """
        super().__init__()
        self.features_size = features_size
        self.label_size = label_size
        self.hierarchy = hierarchy
        self.hidden_size = hidden_size
        self.beta = beta
        self.dropout_rate = dropout_rate

        def local_model(input_dim, hidden_dim, num_labels, dropout_rate):
            return nn.Sequential(
                Dense(input_dim, hidden_dim, activation=nn.ReLU()),
                nn.Dropout(dropout_rate),
                Dense(hidden_dim, num_labels, activation=nn.Sigmoid())
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
        self.sigmoid_model = Dense(hidden_size, label_size, activation=nn.Sigmoid())

    def forward(self, x):
        assert x.shape[-1] == self.features_size
        global_models = []
        local_models = []

        for i in range(len(hierarchy)):
            if i == 0:
                global_models.append(self.global_models[i](x))
            else:
                global_models.append(self.global_models[i](torch.cat([global_models[i - 1], x], dim=1)))
        p_glob = self.sigmoid_model(global_models[-1])

        for i in range(len(hierarchy)):
            local_models.append(self.local_models[i](global_models[i]))

        p_loc = torch.cat(local_models, dim=1)
        labels = (1-self.beta) * p_glob + self.beta * p_loc
        return labels


if __name__=='__main__':

    x = torch.cat([torch.zeros([1000,77]), torch.ones([628,77])], dim=0)
    y = torch.cat([torch.zeros([1000,499]), torch.ones([628,499])], dim=0)

    # (1628, 77)
    # (1628, 499)

    hierarchy = [18, 80, 178, 142, 77, 4]
    feature_size = x.shape[1]
    label_size = y.shape[1]
    beta = 0.5
    model = HMCNFModel(features_size=feature_size, 
                       label_size=label_size, 
                       hierarchy=hierarchy, 
                       hidden_size=384, 
                       beta=0.5, 
                       dropout_rate=0.1)
    y_pred = model(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    for t in range(100):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        # print(y, y_pred)
        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 10 == 9:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_predict = torch.where(y_pred > 0.5, 1, 0)
    y = y.numpy()
    y_predict = y_predict.numpy()
    predict_ok = np.where(np.sum(y_predict - y, axis=1) == 0, 1, 0)
    print("{} good out of {} samples".format(np.sum(predict_ok), predict_ok.shape[0]))