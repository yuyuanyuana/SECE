import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ #
# activation function

exp_act = lambda x: torch.exp(x)

def acti_fun(activate):
    if activate == 'relu':
        return F.relu
    elif activate == 'sigmoid':
        return torch.sigmoid
    elif activate == 'exp':
        return exp_act
    elif activate == 'softplus':
        return F.softplus
    elif activate =='tanh':
        return F.tanh


# ============================================================================ #
# layer

class FC_Layer(nn.Module):
    def __init__(self, in_features, out_features, bn=False, activate='relu', dropout=0.0):
        super(FC_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn
        self.activate = activate
        self.dropout = dropout
        self.layer = nn.Linear(in_features, out_features)
        self.bn_layer = nn.BatchNorm1d(out_features)
    def forward(self, x):
        x = self.layer(x)
        if self.bn:
            x = self.bn_layer(x)
        if self.dropout!=0:
            return F.dropout(acti_fun(self.activate)(x), p=self.dropout, training=self.training)
        return acti_fun(self.activate)(x)       
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# ============================================================================ #
# NB & ZINB model


class New_NB_AE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.2):
        super(New_NB_AE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.layer1 = FC_Layer(self.input_dim, self.hidden_dim, bn=True, activate='relu', dropout=self.dropout)
        self.layer2 = FC_Layer(self.hidden_dim, self.latent_dim, bn=True, activate='relu', dropout=self.dropout)
        self.layer3 = FC_Layer(self.latent_dim, self.hidden_dim, bn=False, activate='relu')
        self.layer4 = FC_Layer(self.hidden_dim, self.input_dim, bn=False, activate='exp')
        self.logits = torch.nn.Parameter(torch.randn(self.input_dim))

    def forward(self, x): 

        z = self.layer2(self.layer1(x))
        rate_scaled = self.layer4(self.layer3(z))

        return rate_scaled, self.logits.exp(), None, z
    


class New_ZINB_AE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.2):
        super(New_ZINB_AE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.layer1 = FC_Layer(self.input_dim, self.hidden_dim, bn=True, activate='relu', dropout=self.dropout)
        self.layer2 = FC_Layer(self.hidden_dim, self.latent_dim, bn=True, activate='relu', dropout=self.dropout)
        self.layer3 = FC_Layer(self.latent_dim, self.hidden_dim, bn=False, activate='relu')
        self.layer_disp = FC_Layer(self.hidden_dim, self.input_dim, bn=False, activate='exp')
        self.layer_drop = FC_Layer(self.hidden_dim, self.input_dim, bn=False, activate='sigmoid')
        self.layer_logi = torch.nn.Parameter(torch.randn(self.input_dim))
    
    def forward(self, x): 

        z = self.layer2(self.layer1(x))
        x3 = self.layer3(z)

        rate_scaled = self.layer_disp(x3)
        dropout = self.layer_drop(x3)

        return rate_scaled, self.layer_logi.exp(), dropout, z
    

model_dict = {'zinb':New_ZINB_AE, 'nb':New_NB_AE}


# ============================================================================ #
# GAT_PYG

from torch_geometric.nn import GATConv

class GAT_pyg(torch.nn.Module):
    def __init__(self, latent_dim=32, dropout_gat=0.5):
        super(GAT_pyg, self).__init__()
        self.dropout = dropout_gat
        self.gat1 = GATConv(latent_dim, latent_dim, 1)
        self.gat2 = GATConv(latent_dim, latent_dim, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        z = F.dropout(x, self.dropout, training=self.training)
        xbar = self.gat2(z, edge_index)
        xbar = F.elu(xbar)
        xbar = F.dropout(xbar, self.dropout, training=self.training)
        return xbar, z




