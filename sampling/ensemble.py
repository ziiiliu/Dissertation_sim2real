import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.basenn import BaseNN

class PSNNUncertainty(BaseNN):
    def __init__(self, n_visible=3, dropout=0.5, n_output=6):
        super(PSNNUncertainty, self).__init__(n_output=6)
        self.n_visible = n_visible
        # Here we alter the dimension in the input layer
        self.input = nn.Linear(self.input_dim * self.n_visible, self.n_hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X):
        res = F.relu(self.input(X))
        for fc in self.fcs:
            res = F.relu(fc(res))
        res = self.dropout(res)
        res = self.predict(res)
        return res

def train(model, )

def predict_with_uncertainty(models, x):
    '''
    Args:
        models: The trained keras model ensemble
        x: the input tensor with shape [N, M]
        samples: the number of monte carlo samples to collect
    Returns:
        y_mean: The expected value of our prediction
        y_std: The standard deviation of our prediction
    '''
    mu_arr = []
    si_arr = []

    for model in models:
        y_pred = model(x)
        mu = y_pred[:, 0]
        si = y_pred[:, 1]

        mu_arr.append(mu)
        si_arr.append(si)

    mu_arr = np.array(mu_arr)
    si_arr = np.array(si_arr)
    var_arr = np.exp(si_arr)

    y_mean = np.mean(mu_arr, axis=0)
    y_variance = np.mean(var_arr + mu_arr**2, axis=0) - y_mean**2
    y_std = np.sqrt(y_variance)
    return y_mean, y_std
