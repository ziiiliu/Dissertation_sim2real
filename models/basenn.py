"""
BaseNN class is meant both for regression-like tasks for modelling the controller
behavior (system identification task), and for uncertainty estimates for sampling purposes.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNN(nn.Module):

    def __init__(self, 
                input_dim=3,
                n_hidden=64,
                n_output=3, 
                n_layer=3,):
        super(BaseNN, self).__init__()
        self.include_past = False
        self.input_dim = input_dim
        self.n_hidden = n_hidden

        # Input Layer
        self.input = nn.Linear(input_dim, n_hidden)

        # Constructing the hidden layers with a modulelist
        self.fcs = []
        for i in range(n_layer):
            self.fcs.append(nn.Linear(n_hidden, n_hidden))
        self.fcs = nn.ModuleList(self.fcs)

        # Prediction Layer
        self.predict = nn.Linear(n_hidden, n_output)
    

