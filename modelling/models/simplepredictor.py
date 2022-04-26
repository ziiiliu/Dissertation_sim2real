"""
SimplePredictor, based on BaseNN, is a MPNN to be trained for approximating the
controller behavior directly. This can be used for predicting the full state or the differential state.
"""


from .basenn import BaseNN
import torch.nn.functional as F
import torch.nn as nn

class SimplePredictor(BaseNN):
    def __init__(self, input_dim, n_hidden, n_output, n_layer, activation=None):
        super(SimplePredictor, self).__init__(
            input_dim=input_dim,
            n_hidden=n_hidden,
            n_output=n_output,
            n_layer=n_layer,
        )
        self.activation = activation
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, X):
        res = F.relu(self.input(X))
        if self.activation is None:
            for fc in self.fcs:
                res = fc(res)
        else:
            for fc in self.fcs:
                res = F.relu(self.dropout(fc(res)))
        res = self.predict(res)
        return res
