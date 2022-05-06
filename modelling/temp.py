
import numpy as np
from models.simplepredictor import SimplePredictor
from utils.utils import get_gleaned_past_state_X
import torch


# model_path = "ckpt/2nd_collect_simplepredictor_differential_0_layer_linear_2D.pt"
# model = SimplePredictor(input_dim=4, n_hidden=64, n_output=2, n_layer=0, activation=None)
# model.load_state_dict(torch.load(model_path))

# print(model)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

cur_states = torch.Tensor(np.load("../second_collection_slower/cur_states.npy"))

input_dim = 2
n_visible = 50
interval = 10

X_ps = torch.Tensor(get_gleaned_past_state_X(cur_states[:, :input_dim], n_visible=n_visible, input_dim=input_dim, interval=interval))

print(cur_states.shape, X_ps.shape)