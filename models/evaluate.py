from simplepredictor import SimplePredictor
import torch
import numpy as np
from utils import train_val_test_split

def evaluate_from_origin(model, X_val, y_val, steps=1000):
    """
    This evaluate function takes only the first (few) inputs and the single prediction result will be
    the input for the next evaluation step. We can therefore see how the error propagates as the trained model
    runs for longer.
    """
    results = []
    cur_state = (X_val[0])
    results.append(cur_state.detach().numpy()[:3])
    for i in range(1, steps):
        next_state = model(cur_state)
        results.append(next_state.detach().numpy())
        cur_state = torch.cat([next_state, X_val[i,3:]])
        print(cur_state)
    return results

if __name__ == "__main__":

    cur_states = torch.Tensor(np.load("../first_collection/cur_states.npy"))
    ref_states = torch.Tensor(np.load("../first_collection/ref_states.npy"))
    model_path = "ckpt/best_simplepredictor_2D.pt"

    model = SimplePredictor(input_dim=6, n_hidden=64, n_output=3, n_layer=3)
    model.load_state_dict(torch.load(model_path))

    X = torch.cat([cur_states, ref_states], axis=1)[:-1]
    y = cur_states[1:]

    steps = 5000
    eval_results = evaluate_from_origin(model, X, y, steps=steps)
    # print(eval_results)
    with open(f"data/simplepredictor_{steps}_steps_from_start.npy", "wb") as f:
        np.save(f, np.asarray(eval_results))
    