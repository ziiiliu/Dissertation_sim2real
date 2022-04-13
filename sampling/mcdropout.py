import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models.basenn import BaseNN
from baal.active import ActiveLearningDataset
import copy

from utils import set_seed, train_val_test_split
from metrics import get_batch_bald
from dataset import VelDataset


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class PSNNUncertainty(BaseNN):
    def __init__(self, n_visible=3, dropout=0.5, n_output=6):
        super(PSNNUncertainty, self).__init__(n_output=6)
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

            loss = loss_func(predictions, target.double())
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
            # scheduler.step()
            train_loss += loss.item() * target.shape[0]

        if epoch % 10 == 0:
            writer.add_scalar('train_loss', loss, global_step=epoch)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        print(f'epoch number: {epoch+1}, MSE Loss: {loss.data}')
        
        
        if epoch % 100 == 0:
            for batch in test_loader:
                val_feature, val_target = batch
                val_feature, val_target = val_feature.to(device), val_target.to(device)
                val_predictions = model(val_feature)

                val_loss = loss_func(val_predictions, val_target)
            writer.add_scalar('validation_loss', val_loss, global_step=epoch)
            print('Validation Loss: ', val_loss.data)
            val_losses.append(val_loss.data.item())
            # if val_loss.data < best_val_loss:
            #     best_val_loss = val_loss.data
            #     torch.save(model.state_dict(), model_save_path)

    return train_losses, val_losses

#-------------------------Monte Carlo Dropout Scripts---------------------------#
def enable_dropout(model):
    # Enable dropout layers at test time because by default they only work in training
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_preds_labels_rnn_mc(model, data_loader, ensemble_size, rounded_preds=True):
    # NB: Don't worry about the labels because we'll sort them out when combining in cross-val
    model.eval()
    enable_dropout(model)
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for ens in range(ensemble_size):
            # predictions for a single model in the ensemble
            predictions = np.array([])
            for batch in data_loader:
                feature, lengths, target = batch
                feature, target = feature.to(device), target.to(device)
                batch_preds = model(feature, lengths)
                predictions = np.append(predictions, batch_preds.detach().cpu().numpy())
                if ens == 0:
                    all_labels.extend(target.detach().cpu().numpy())
            all_predictions.append(np.array(predictions))
    # all_predictions.shape = (ensemble_size, data_size)
    all_predictions = np.stack(all_predictions)
    # all_labels.shape = (data_size)
    all_labels = np.array(all_labels)
    if rounded_preds:
        return np.rint(all_predictions), all_labels
    return all_predictions, all_labels

#------------------------------------------------------------------------------#

