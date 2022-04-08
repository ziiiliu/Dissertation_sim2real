from utils import train_val_test_split, get_past_state_X
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as MSE
import numpy as np

if __name__ == "__main__":
    n_visible = 5

    cur_states = np.load("../first_collection/cur_states.npy")
    ref_states = np.load("../first_collection/ref_states.npy")

    X_ps = get_past_state_X(cur_states, n_visible=n_visible)
    X = np.concatenate([X_ps, ref_states[n_visible-1:-1]], axis=1)
    y = cur_states[n_visible:]

    print(X_ps.shape, y.shape, X.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    rf = RandomForestRegressor()
    rf.fit(X_train,y_train)

    test_preds = rf.predict(X_test)
    loss = MSE(y_test, test_preds)
    print(loss)
