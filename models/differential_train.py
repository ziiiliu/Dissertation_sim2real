from simplepredictor import SimplePredictor
from paststates_nn import PSNN

from utils import train_val_test_split, get_past_state_X
import torch
import torch.optim as optim
import numpy as np

def train_differential(model, X_train, y_train, X_val, y_val, 
        epochs=1000, model_save_path="ckpt/best_simplepredictor.pt",
        lr=1e-4, opt='adam'):
    if opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer choice unknown")
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    loss_func = torch.nn.MSELoss()

    losses = []

    for epoch in range(epochs):
    
        best_val_loss = float('inf')
        
        prediction = model(X_train)
        loss = loss_func(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()         
        optimizer.step()
    #     scheduler.step()
        print(f'epoch number: {epoch+1}, MSE Loss: {loss.data}')
        
        if epoch % 100 == 0:
            val_y_preds = model(X_val)
            val_loss = loss_func(val_y_preds, y_val)
            print('Validation Loss: ', val_loss.data)
            losses.append(val_loss.data.item())
            if val_loss.data < best_val_loss:
                best_val_loss = val_loss.data
                torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":

    # model = SimplePredictor(input_dim=6, n_hidden=64, n_output=3, n_layer=3)
    n_visible = 50
    model = PSNN(n_visible=n_visible)
    epochs = 3000
    # model_path = "ckpt/best_simplepredictor_5_layer_2D.pt"
    model_path = "ckpt/best_psnn_50_visible_differential.pt"

    cur_states = torch.Tensor(np.load("../first_collection/cur_states.npy"))
    ref_states = torch.Tensor(np.load("../first_collection/ref_states.npy"))

    # print(cur_states[400:600], ref_states[400:600])

    X_ps = torch.Tensor(get_past_state_X(cur_states, n_visible=n_visible))
    # X = torch.cat([cur_states, ref_states], axis=1)[:-1]
    X = torch.cat([X_ps, ref_states[n_visible-1:-1]], axis=1)
    y = torch.diff(cur_states, dim=0)[n_visible-1:]

    print(X_ps.shape, y.shape, X.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    train_differential(model, X_train, y_train, X_val, y_val, epochs = epochs, model_save_path=model_path, lr=1e-3, opt="adam")