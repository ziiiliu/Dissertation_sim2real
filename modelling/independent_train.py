from models.simplepredictor import SimplePredictor
from models.paststates_nn import PSNN
from torch.nn import LSTM


from utils.utils import train_val_test_split, get_past_state_X
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
import os

from torch.utils.tensorboard import SummaryWriter

def train_independent(models, X_train, y_train, X_val, y_val, 
        epochs=1000, model_save_path="ckpt/best_simplepredictor.pt",
        lr=1e-4, opt='adam', writer=None):
    
    for i, model in enumerate(models):
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
            
            prediction = model(X_train[:,i::3])
            # print(X_train[0], y_train[0], prediction[0])
            loss = loss_func(prediction, torch.unsqueeze(y_train[:, i], dim=1))

            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        #     scheduler.step()
            if epoch % 10 == 0:
                writer.add_scalar(f'train_loss_{i}', loss, global_step=epoch)
            print(f'epoch number: {epoch+1}, MSE Loss: {loss.data}')
            
            if epoch % 100 == 0:
                val_y_preds = model(X_val[:, i::3])
                val_loss = loss_func(val_y_preds, torch.unsqueeze(y_val[:, i], dim=1))
                writer.add_scalar(f'validation_loss_{i}', val_loss, global_step=epoch)
                print('Validation Loss: ', val_loss.data)
                losses.append(val_loss.data.item())
                if val_loss.data < best_val_loss:
                    best_val_loss = val_loss.data
                    if model_save_path is not None:
                        torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":

    # model = SimplePredictor(input_dim=4, n_hidden=64, n_output=2, n_layer=1, activation=None)
    # print(model)
    n_visible = 100
    model1 = PSNN(n_visible=n_visible, n_output=1, n_layer=3, input_dim=1)
    model2 = PSNN(n_visible=n_visible, n_output=1, n_layer=3, input_dim=1)
    models = [model1, model2]
    
    epochs = 10000
    # num_samples = 1000

    # lag_offset shifts the y to account for the lag of response of the real robot
    lag_offset = 0

    # model_path = "ckpt/best_simplepredictor_differential_1_layer_linear_2D_1000_samples.pt"
    model_path = "ckpt/best_psnn_100_visible_differential_independent.pt"

    cur_states = torch.Tensor(np.load("../first_collection/cur_states.npy"))
    ref_states = torch.Tensor(np.load("../first_collection/ref_states.npy"))

    # print(cur_states[400:600], ref_states[400:600])

    X_ps = torch.Tensor(get_past_state_X(cur_states, n_visible=n_visible))
    if lag_offset == 0:
        X = torch.cat([cur_states[:, :-1], ref_states[:, :-1]], axis=1)[:-1]
    else:
        X = torch.cat([cur_states[lag_offset:, :-1], ref_states[:-lag_offset, :-1]], axis=1)[:-1]

    X = torch.cat([X_ps, ref_states[n_visible-1:-1]], axis=1)
    # y = torch.diff(cur_states, dim=0)[n_visible-1:]
    y = torch.diff(cur_states[lag_offset:, :-1], dim=0)[n_visible-1:]

    print(y.shape, X.shape)
    # print(X[:50], y[:50])

    # setting up tensorboard writer and logging
    dir_name = model_path[10:-3] + datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('log', dir_name)
    writer = SummaryWriter(log_dir=log_dir)

    # X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X[:num_samples], y[:num_samples])
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    train_independent(models, X_train, y_train, X_val, y_val, epochs=epochs, model_save_path=None, lr=5e-4, opt="adam", writer=writer)