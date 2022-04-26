"""
Paststates NN model incorporates information from previous N timesteps.
"""
from .basenn import BaseNN
import torch.nn as nn
import torch.nn.functional as F

class PSNN(BaseNN):
    """
    Params
    ----------
    n_visible:  int, number of past timesteps to pass into the model (including the current timestep).
                n = 1 reduces this model to SimplePredictor
    """
    def __init__(self, n_visible=3, n_output=3, n_layer=3, input_dim=3):
        super(PSNN, self).__init__(n_output=n_output, n_layer=n_layer, input_dim=input_dim)
        self.n_visible = n_visible
        # Here we alter the dimension in the input layer
        self.input = nn.Linear(self.input_dim * (self.n_visible+1), self.n_hidden)

    def forward(self, X):
        res = F.relu(self.input(X))
        for fc in self.fcs:
            res = F.relu(fc(res))
        res = self.predict(res)
        return res