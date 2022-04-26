"""
Ensemble uncertainty estimation 
adapted from https://github.com/huyng/incertae/blob/master/ensemble_regression.ipynb

Paper: https://arxiv.org/abs/1612.01474 

Active learning scripts adapted from 
https://github.com/rampopat/uncertainty/blob/main/Main_Notebook_DementiaBank.ipynb


"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from baal.active import ActiveLearningDataset
import copy

from utils import set_seed, train_val_test_split
from metrics import get_batch_bald, get_expected_entropy, get_mutual_info, get_pred_var
from dataset import VelDataset


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class BaseNN(nn.Module):

    def __init__(self, 
                input_dim=3,
                n_hidden=64,
                n_output=3, 
                n_layer=3,):
        super(BaseNN, self).__init__()
        self.include_past = False
        self.input_dim = input_dim
        self.n_hidden = n_hidden

        # Input Layer
        self.input = nn.Linear(input_dim, n_hidden)

        # Constructing the hidden layers with a modulelist
        self.fcs = []
        for i in range(n_layer):
            self.fcs.append(nn.Linear(n_hidden, n_hidden))
        self.fcs = nn.ModuleList(self.fcs)
        # Prediction Layer
        self.predict = nn.Linear(n_hidden, n_output)


class PSNNUncertainty(BaseNN):
    def __init__(self, n_visible=3, dropout=0.5, n_output=6, n_layer=3, input_dim=3):
        super(PSNNUncertainty, self).__init__(n_output=n_output, n_layer=n_layer, input_dim=input_dim)
        self.n_visible = n_visible
        # Here we alter the dimension in the input layer
        self.input = nn.Linear(self.input_dim * self.n_visible, self.n_hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X):
        res = F.relu(self.input(X))
        for fc in self.fcs:
            res = F.relu(fc(res))
        res = self.dropout(res)
        res = self.predict(res)
        return res


def get_preds_labels(model, data_loader, rounded_preds=True):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            feature, target = batch
            feature, target = feature.to(device), target.to(device)
            predictions = model(feature)
            # Edge case when batch size = 1 happens because model squeezes
            if predictions.shape == ():
                # No need to unsqueeze target because model is squeezing
                predictions = predictions.unsqueeze(0)
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(target.detach().cpu().numpy())
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    if rounded_preds:
        return np.rint(all_predictions), all_labels
    return all_predictions, all_labels

def train_differential_psnn(model, train_loader, test_loader, criterion=None,
                        optimizer='adam', device=None, epochs=1000, lr=1e-4, 
                        opt='adam', writer=None):
    if opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer choice unknown")
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    loss_func = torch.nn.MSELoss()

    val_losses = []
    train_losses = []

    for epoch in range(epochs):
        train_loss = 0
        best_val_loss = float('inf')

        for batch in train_loader:
            feature, target = batch
            feature, target = feature.to(device), target.to(device)

            predictions = model(feature)
            optimizer.zero_grad()

            loss = loss_func(predictions, target)
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
            # scheduler.step()
            train_loss += loss.item() * target.shape[0]

        if epoch % 10 == 0 and writer is not None:
            writer.add_scalar('train_loss', loss, global_step=epoch)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        # print(f'epoch number: {epoch+1}, MSE Loss: {loss.data}') 
        
        if epoch % 100 == 0:
            for batch in test_loader:
                val_feature, val_target = batch
                val_feature, val_target = val_feature.to(device), val_target.to(device)
                val_predictions = model(val_feature)

                val_loss = loss_func(val_predictions, val_target)
            if writer is not None:
                writer.add_scalar('validation_loss', val_loss, global_step=epoch)
            print('Validation Loss: ', val_loss.data)
            val_losses.append(val_loss.data.item())
            # if val_loss.data < best_val_loss:
            #     best_val_loss = val_loss.data
            #     torch.save(model.state_dict(), model_save_path)

    return train_losses, val_losses

def predict_with_uncertainty(models, x):
    '''
    Args:
        models: The trained keras model ensemble
        x: the input tensor with shape [N, M]
        samples: the number of monte carlo samples to collect
    Returns:
        y_mean: The expected value of our prediction
        y_std: The standard deviation of our prediction
    '''
    mus_arr = []
    sigs_arr = []

    for model in models:
        y_pred = model(x)
        mu = y_pred[:, 0]
        si = y_pred[:, 1]

        mus_arr.append(mu)
        sigs_arr.append(si)

    mu_arr = np.array(mu_arr)
    si_arr = np.array(si_arr)
    var_arr = np.exp(si_arr)

    y_mean = np.mean(mu_arr, axis=0)
    y_variance = np.mean(var_arr + mu_arr**2, axis=0) - y_mean**2
    y_std = np.sqrt(y_variance)
    return y_mean, y_std


def active_ensemble(train_data, test_data, heuristic=None, ensemble_size=10, init_seed=1, 
                            reset_weights=False, init_train_size=75, ndata_to_label=40, 
                            al_steps=None, num_epochs=16, vel_dim=2):

    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4

    # Compute number of active learning steps based on ndata_to_label
    if not al_steps:
        al_steps = int(np.ceil((len(train_data) - init_train_size) / ndata_to_label)) + 1
    active_train = ActiveLearningDataset(dataset=train_data, labelled=None)
    active_train.label_randomly(init_train_size)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Initialize the models in the ensemble
    ensemble_weights = []
    ensemble_opts = []
    for seed_offset in range(ensemble_size):
        seed = init_seed + seed_offset
        set_seed(seed)
        model = PSNNUncertainty(n_visible=n_visible, n_output=2, n_layer=3, input_dim=4)
        model_weights = copy.deepcopy(model.state_dict())    
        ensemble_weights.append(model_weights)    
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        ensemble_opts.append(optimizer.state_dict())
    
    active_ensemble_test_preds = np.zeros((al_steps, ensemble_size, len(test_data), vel_dim))
    train_sizes = []
    test_labels = test_data.y

    # Active learning loop starts here
    for al_step in range(al_steps-1):
        train_sizes.append(active_train.n_labelled)

        # Loading the training and pool data
        active_train_loader = DataLoader(active_train, batch_size=BATCH_SIZE)
        active_pool_loader = DataLoader(active_train.pool, batch_size=BATCH_SIZE)

        ensemble_pool_preds = np.zeros((ensemble_size, active_train.n_unlabelled, vel_dim))

        for seed_offset in range(ensemble_size):
            print('AL STEP: {}/{}'.format(al_step + 1, al_steps), 'Model: {}/{}'.format(seed_offset + 1, ensemble_size), 
                '| Train Size:', active_train.n_labelled, '| Pool Size:', active_train.n_unlabelled)
            seed = init_seed + seed_offset 
            set_seed(seed)

            model.load_state_dict(ensemble_weights[seed_offset])       
            optimizer.load_state_dict(ensemble_opts[seed_offset]) 
            train_losses, val_losses = train_differential_psnn(model, active_train_loader, test_loader, epochs=num_epochs, optimizer=optimizer, 
                                             device=device)

            # in the reset_weights case we simply load in the initial weights each time
            if not reset_weights:
                ensemble_weights[seed_offset] = copy.deepcopy(model.state_dict())
                ensemble_opts[seed_offset] = copy.deepcopy(optimizer.state_dict())
        
            pool_preds, _ = get_preds_labels(model, active_pool_loader, rounded_preds=False)

            for i in range(vel_dim):
                ensemble_pool_preds[seed_offset, :, i] = pool_preds[:, i]

            # Evaluate model on test set 
            test_preds, _ = get_preds_labels(model, test_loader, rounded_preds=False)
            for i in range(vel_dim):
                active_ensemble_test_preds[al_step, seed_offset, :, i] = test_preds[:, i]
        
        # Get pool set uncertainties to determine labelling
        if heuristic is None: # random labelling
            active_train.label_randomly(min(ndata_to_label, active_train.n_unlabelled))
        elif active_train.n_unlabelled > ndata_to_label: 
            if heuristic == get_batch_bald:
                uncertain_indices = get_batch_bald(ensemble_pool_preds, ndata_to_label)
            else:
                uncertainties_pool = heuristic(ensemble_pool_preds)
                print(uncertainties_pool)
                uncertainties_pool = np.linalg.norm(uncertainties_pool, axis=1)
                print(ensemble_pool_preds)
                # print(uncertainties_pool.shape)
                print(uncertainties_pool)
                uncertain_indices = np.argpartition(uncertainties_pool, -ndata_to_label)[-ndata_to_label:]
            active_train.label(uncertain_indices)
        else: # last al_step is to label remainder of pool < ndata_to_label
            active_train.label(np.arange(active_train.n_unlabelled))

    train_sizes = np.array(train_sizes) / len(train_data)
    return active_ensemble_test_preds, train_sizes


if __name__ == "__main__":

    n_visible = 1
    vel_dim = 2

    cur_states = torch.Tensor(np.load("../first_collection/cur_states.npy"))
    ref_states = torch.Tensor(np.load("../first_collection/ref_states.npy"))
    X = torch.cat([cur_states[:, :-1], ref_states[:, :-1]], axis=1)[:-1]
    y = torch.diff(cur_states[:, :-1], dim=0)[n_visible-1:]

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X[:1000], y[:1000])
    train_data = VelDataset(X_train, y_train)
    test_data = VelDataset(X_val, y_val)

    active_ensemble(train_data, test_data, heuristic=None, ensemble_size=5, init_seed=1, 
                            reset_weights=False, init_train_size=75, ndata_to_label=40, 
                            al_steps=None, num_epochs=100, vel_dim=vel_dim)