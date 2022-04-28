from models.simplepredictor import SimplePredictor
from models.paststates_nn import PSNN
import torch
import numpy as np
from utils.utils import train_val_test_split, get_past_state_X

def evaluate_from_origin(model, X_val, y_val, steps=1000, n_visible=1):
    """
    This evaluate function takes only the first (few) inputs and the single prediction result will be
    the input for the next evaluation step. We can therefore see how the error propagates as the trained model
    runs for longer.
    """
    results = []
    cur_state = X_val[0]
    results.append(cur_state.detach().numpy()[:3])
    print(cur_state.shape)
    for i in range(1, steps):
        next_state = model(cur_state)
        results.append(next_state.detach().numpy())
        cur_state = torch.cat([cur_state[3:-3], next_state, X_val[i,3*n_visible:]])
    return results

def evaluate_differential_from_origin(model, X_val, y_val, steps=1000, n_visible=1, input_dim=2):
    results = []
    cur_state = X_val[0]
    results.append(cur_state.detach().numpy()[:input_dim])
    print(cur_state.shape)
    for i in range(1, steps):
        diff_state = model(cur_state)
        next_state = cur_state[-input_dim*2:-input_dim] + diff_state
        results.append(next_state.detach().numpy())
        cur_state = torch.cat([cur_state[input_dim:-input_dim], next_state, X_val[i,input_dim*n_visible:]])
        print(i, diff_state, cur_state)
    return results


if __name__ == "__main__":

    include_past = True
    differential = True
    n_visible=100

    # cur_states = torch.Tensor(np.load("../first_collection/cur_states.npy"))
    # ref_states = torch.Tensor(np.load("../first_collection/ref_states.npy"))
    # model_path = "ckpt/best_simplepredictor_differential_1_layer_linear_2D_1000_samples_shift_50.pt"

    cur_states = torch.Tensor(np.load("../second_collection_slower/cur_states.npy"))
    ref_states = torch.Tensor(np.load("../second_collection_slower/ref_states.npy"))
    model_path = "ckpt/2nd_collect_psnn_50_visible_differential.pt"

    # model = SimplePredictor(input_dim=4, n_hidden=64, n_output=2, n_layer=1)
    # model = PSNN(n_visible=n_visible, n_output=3, n_layer=3)
    # model.load_state_dict(torch.load(model_path))

    lag_shift = 0

    if lag_shift == 0:
        X = torch.cat([cur_states[:, :-1], ref_states[:, :-1]], axis=1)[:-1]
    else:
        X = torch.cat([cur_states[lag_shift:, :-1], ref_states[:-lag_shift, :-1]], axis=1)[:-1]
    y = cur_states[1:, :-1]

    X_ps = torch.Tensor(get_past_state_X(cur_states, n_visible=n_visible))
    X = torch.cat([X_ps, ref_states[n_visible-1:-1]], axis=1)
    y = torch.diff(cur_states, dim=0)[n_visible-1:]
    print(y.shape, X.shape)

    if include_past:
        if differential:
            n_visible = 100
            model_path = "ckpt/2nd_collect_psnn_100_visible_differential.pt"
            model = PSNN(n_visible=n_visible, n_output=3, n_layer=3)
            model.load_state_dict(torch.load(model_path))
            X_ps = torch.Tensor(get_past_state_X(cur_states, n_visible=n_visible))

            X = torch.cat([X_ps, ref_states[n_visible-1:-1]], axis=1)
            y = torch.diff(cur_states, dim=0)[n_visible-1:]

        else:
            n_visible=5
            model_path = "ckpt/best_psnn_5_visible.pt"
            model = PSNN(n_visible=n_visible)
            model.load_state_dict(torch.load(model_path))
            X_ps = torch.Tensor(get_past_state_X(cur_states, n_visible=n_visible))
            X = torch.cat([X_ps, ref_states[n_visible-1:-1]], axis=1)
            y = cur_states[n_visible:]

    steps = 10000
    if differential:
        eval_results = evaluate_differential_from_origin(model, X, y, steps=steps, n_visible=n_visible, input_dim=3)
    else:
        eval_results = evaluate_from_origin(model, X, y, steps=steps, n_visible=n_visible)
    # print(eval_results)
    # with open(f"data/simplepredictor_1_layer_linear_differential_{steps}_steps_shift_50.npy", "wb") as f:
    #     np.save(f, np.asarray(eval_results))
    with open(f"data/2nd_psnn_visible_100_differential_{steps}_steps.npy", "wb") as f:
        np.save(f, np.asarray(eval_results))