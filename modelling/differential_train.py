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

def print_model(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

def train_differential(model, X_train, y_train, X_val, y_val, 
        epochs=1000, model_save_path="ckpt/best_simplepredictor.pt",
        lr=1e-4, opt='adam', writer=None):
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
        # print(X_train[0], y_train[0], prediction[0])
        loss = loss_func(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()         
        optimizer.step()
    #     scheduler.step()
        if epoch % 10 == 0:
            writer.add_scalar('train_loss', loss, global_step=epoch)
        print(f'epoch number: {epoch+1}, MSE Loss: {loss.data}')
        
        if epoch % 100 == 0:
            val_y_preds = model(X_val)
            val_loss = loss_func(val_y_preds, y_val)
            writer.add_scalar('validation_loss', val_loss, global_step=epoch)
            print('Validation Loss: ', val_loss.data)
            losses.append(val_loss.data.item())
            if val_loss.data < best_val_loss:
                best_val_loss = val_loss.data
                torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":

    # model = SimplePredictor(input_dim=4, n_hidden=64, n_output=2, n_layer=0, activation=None)
    # print_model(model)


    n_visible = 10
    input_dim = 2

    model = PSNN(n_visible=n_visible, input_dim=input_dim, n_output=input_dim, n_layer=3)
    
    epochs = 5000
    # num_samples = 1000

    # lag_offset shifts the y to account for the lag of response of the real robot
    lag_offset = 50

    # model_path = "ckpt/2nd_collect_simplepredictor_differential_0_layer_linear_2D.pt"
    # model_path = "ckpt/best_psnn_100_visible_differential.pt"
    model_path = "ckpt/2nd_collect_psnn_10_visible_differential_shift_50.pt"
    # model_path = "ckpt/2nd_collect_simple_differential_1_layer_linear_2D.pt"

    cur_states = torch.Tensor(np.load("../second_collection_slower/cur_states.npy"))
    ref_states = torch.Tensor(np.load("../second_collection_slower/ref_states.npy"))
    data_size = len(ref_states)

    # print(cur_states[400:600], ref_states[400:600])

    if n_visible > 1:
        X_ps = torch.Tensor(get_past_state_X(cur_states[:, :input_dim], n_visible=n_visible, input_dim=input_dim))
        X = torch.cat([X_ps[lag_offset:], ref_states[n_visible-1:data_size-lag_offset-1, :input_dim]], axis=1)
    elif n_visible == 1:
        X = torch.cat([cur_states[lag_offset:, :input_dim], ref_states[:data_size-lag_offset, :input_dim]], axis=1)[:-1]
    else:
        raise ValueError("n_visible defined incorrectly")

    y = torch.diff(cur_states[lag_offset:, :input_dim], dim=0)[n_visible-1:]

    print(y.shape, X.shape)
    # print(X[:50], y[:50])

    # setting up tensorboard writer and logging
    dir_name = model_path[5:-3] + datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('log', dir_name)
    writer = SummaryWriter(log_dir=log_dir)

    # X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X[:num_samples], y[:num_samples])
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    train_differential(model, X_train, y_train, X_val, y_val, epochs=epochs, model_save_path=model_path, lr=5e-4, opt="adam", writer=writer) 