from utils import get_past_state_X
import numpy as np

cur_states = np.load("../first_collection/cur_states.npy")
X_ps = get_past_state_X(cur_states)
print(X_ps.shape)
