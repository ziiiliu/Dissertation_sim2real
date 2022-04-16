
import torch.nn.functional as F
from torch.nn import LSTM
import torch
import torch.optim as optim
import numpy as np

import os
from datetime import datetime

from utils import get_past_state_X, train_val_test_split
from torch.utils.tensorboard import SummaryWriter

def train_rnn_differential(model, X_train, y_train, X_val, y_val, 
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
        
        prediction = torch.squeeze(model(X_train)[1][0])
        loss = loss_func(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()         
        optimizer.step()
    #     scheduler.step()
        # Write to tensorboard
        if epoch % 10 == 0:
            writer.add_scalar('train_loss', loss, global_step=epoch)
        print(f'epoch number: {epoch+1}, MSE Loss: {loss.data}')
        
        if epoch % 100 == 0:
            val_y_preds = torch.squeeze(model(X_val)[1][0])

            val_loss = loss_func(val_y_preds, y_val)
            writer.add_scalar('validation_loss', val_loss, global_step=epoch)
            print('Validation Loss: ', val_loss.data)
            losses.append(val_loss.data.item())
            if val_loss.data < best_val_loss:
                best_val_loss = val_loss.data
                torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":

    # n_visible here is equivalent to the sequence length, 
    n_visible = 10
    epochs = 3000

    rnn = LSTM(input_size=3, hidden_size=32, num_layers=3, batch_first=True, proj_size=3)
    model_path = "ckpt/best_3_layer_lstm_10_visible.pt"

    cur_states = torch.Tensor(np.load("../first_collection/cur_states.npy"))
    ref_states = torch.Tensor(np.load("../first_collection/ref_states.npy"))

    # print(cur_states[400:600], ref_states[400:600])

    X_ps = torch.Tensor(get_past_state_X(cur_states, n_visible=n_visible))
    # X = torch.cat([cur_states, ref_states], axis=1)[:-1]
    X = torch.cat([X_ps, ref_states[n_visible-1:-1]], axis=1)
    y = cur_states[n_visible:]

    X = X.reshape((len(X), n_visible+1, 3))

    print(X_ps.shape, y.shape, X.shape)

    # setting up tensorboard writer and logging
    dir_name = model_path[10:-3] + datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('log', dir_name)
    writer = SummaryWriter(log_dir=log_dir)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    train_rnn_differential(rnn, X_train, y_train, X_val, y_val, epochs = epochs, model_save_path=model_path, lr=5e-3, opt="adam", writer=writer)