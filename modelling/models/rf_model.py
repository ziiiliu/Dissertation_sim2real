from utils import train_val_test_split, get_past_state_X
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as MSE
from joblib import dump
import numpy as np

if __name__ == "__main__":
    n_visible = 10
    input_dim = 2
    lag_offset = 0

    cur_states = np.load("../second_collection_slower/cur_states.npy")
    ref_states = np.load("../second_collection_slower/ref_states.npy")
    data_size = len(ref_states)

    X_ps = get_past_state_X(cur_states[:, :input_dim], n_visible=n_visible, input_dim=input_dim)
    X = np.concatenate([X_ps[lag_offset:], ref_states[n_visible-1:data_size-lag_offset-1, :input_dim]], axis=1)

    y = np.diff(cur_states[lag_offset:, :input_dim], axis=0)[n_visible-1:]

    print(X_ps.shape, y.shape, X.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    rf = RandomForestRegressor()
    rf.fit(X_train,y_train)

    dump(rf, "base_ckpt/2nd_rf_10_visible_differential.pt")

    test_preds = rf.predict(X_test)
    loss = MSE(y_test, test_preds)
    print(loss)
