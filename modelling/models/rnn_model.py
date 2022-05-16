
from turtle import forward
import torch.nn.functional as F
from torch.nn import LSTM, RNN
import torch
import torch.optim as optim
import numpy as np

import os
from datetime import datetime

from utils import get_past_state_X, train_val_test_split
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers=3, batch_first=True, proj_size=2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = RNN(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.output = torch.nn.Linear(hidden_size, proj_size)
    
    def forward(self, X):

        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).requires_grad_().to(device)
        out, h0 = self.rnn(X, h0)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        res = self.output(out)
        return res


def train_rnn_differential(model, X_train, y_train, X_val, y_val, 
        epochs=1000, model_save_path="ckpt_may/best_simplepredictor.pt",
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
        res = model(X_train)
        print(res.shape)
        prediction = torch.squeeze(res[0][:, -1, :])
        print(prediction.shape)
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
    input_dim = 2
    lag_offset = 0
    interval = 5
    

    gleaning=False

    # rnn = RNNModel(input_dim=input_dim, hidden_size=32, num_layers=3, batch_first=True, proj_size=input_dim)
    rnn = LSTM(input_size=input_dim, hidden_size=32, num_layers=3, batch_first=True, proj_size=input_dim)
    rnn.to(device)
    model_path = "ckpt_may/3_layer_lstm_10_visible.pt"

    cur_states = torch.Tensor(np.load("../second_collection_slower/cur_states_smoothed.npy"))
    ref_states = torch.Tensor(np.load("../second_collection_slower/ref_states.npy"))
    data_size = len(ref_states)

    # print(cur_states[400:600], ref_states[400:600])

    if n_visible > 1:
        if gleaning:
            X_ps = torch.Tensor(get_gleaned_past_state_X(cur_states[:, :input_dim], n_visible=n_visible, input_dim=input_dim, interval=interval))
        else:
            X_ps = torch.Tensor(get_past_state_X(cur_states[:, :input_dim], n_visible=n_visible, input_dim=input_dim))
        X = torch.cat([X_ps[lag_offset:], ref_states[n_visible-1:data_size-lag_offset-1, :input_dim]], axis=1)
    elif n_visible == 1:
        X = torch.cat([cur_states[lag_offset:, :input_dim], ref_states[:data_size-lag_offset, :input_dim]], axis=1)[:-1]
    else:
        raise ValueError("n_visible defined incorrectly")

    y = torch.diff(cur_states[lag_offset:, :input_dim], dim=0)[n_visible-1:]

    X = X.reshape((len(X), n_visible+1, input_dim))
    X, y = X.to(device), y.to(device)

    print(X_ps.shape, y.shape, X.shape)

    # setting up tensorboard writer and logging
    dir_name = model_path[10:-3] + datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('log_may', dir_name)
    writer = SummaryWriter(log_dir=log_dir)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    train_rnn_differential(rnn, X_train, y_train, X_val, y_val, epochs = epochs, model_save_path=model_path, lr=5e-3, opt="adam", writer=writer)