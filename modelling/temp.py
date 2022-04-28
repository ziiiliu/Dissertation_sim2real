
import numpy as np
from models.simplepredictor import SimplePredictor
import torch


model_path = "ckpt/2nd_collect_simplepredictor_differential_0_layer_linear_2D.pt"
model = SimplePredictor(input_dim=4, n_hidden=64, n_output=2, n_layer=0, activation=None)
model.load_state_dict(torch.load(model_path))

print(model)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)