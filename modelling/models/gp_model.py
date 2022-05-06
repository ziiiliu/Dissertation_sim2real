"""
GPModel provides a baseline for evaluation of NN-based learned simulators.
ref: https://github.com/cornellius-gp/gpytorch/blob/master/examples/03_Multitask_Exact_GPs/ModelList_GP_Regression.ipynb

"""
import gpytorch
import torch
from torch.nn import MSELoss
import numpy as np
from utils.utils import get_past_state_X, train_val_test_split  

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == "__main__":

    ## TODO: existing issue: cannot take more than 20,000 samples due to a outofmemory error. 
    # Does not scale well. Each epoch takes around 4 seconds for 10000 samples. Rather slow.

    n_visible = 10
    input_dim = 2
    lag_offset = 0
    epochs = 200

    cur_states = torch.Tensor(np.load("../second_collection_slower/cur_states.npy"))
    ref_states = torch.Tensor(np.load("../second_collection_slower/ref_states.npy"))
    data_size = len(ref_states)
    
    # print(cur_states[400:600], ref_states[400:600])

    X_ps = torch.Tensor(get_past_state_X(cur_states[:, :input_dim], n_visible=n_visible, input_dim=input_dim))
    X = torch.cat([X_ps[lag_offset:], ref_states[n_visible-1:data_size-lag_offset-1, :input_dim]], axis=1)

    y = torch.diff(cur_states[lag_offset:, :input_dim], dim=0)[n_visible-1:]

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    print(X_train.shape)

    likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
    model1 = GPModel(X_train[:10000, ::input_dim], y_train[:10000, 0], likelihood1)

    likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
    model2 = GPModel(X_train[:10000, 1::input_dim], y_train[:10000, 1], likelihood2)

    model = gpytorch.models.IndependentModelList(model1, model2)
    likelihood = gpytorch.likelihoods.LikelihoodList(model1.likelihood, model2.likelihood)
    model.train()
    likelihood.train()

    print(model.train_inputs)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model)

    for i in range(epochs):
    # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(*model.train_inputs)
        # Calc loss and backprop gradients
        loss = -mll(output, model.train_targets)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (
            i + 1, epochs, loss.item(),
        ))
        optimizer.step()
    
    torch.save(model.state_dict(), "base_ckpt/2nd_gp_10_visible_differential.pth")

    # Evaluate the trained model
    model.eval()
    likelihood.eval()
    
    f_preds = model(X_test)
    y_preds = likelihood(model(X_test))

    loss_func = MSELoss()
    print(loss_func(f_preds, y_test))

    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix
    f_samples = f_preds.sample(sample_shape=torch.Size(1000,))