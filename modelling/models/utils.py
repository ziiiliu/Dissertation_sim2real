from cmath import nan
import pandas as pd
import numpy as np
from datetime import datetime

def read_csv(cur_state_file, ref_state_file):
    
    cur_states = pd.read_csv(cur_state_file)
    cur_states['timestamp'] = cur_states['time'].apply(lambda t: datetime.strptime(t[:-3], "%Y/%m/%d %H:%M:%S.%f"))
    cur_states['ref_state'] = np.nan
    cur_states = cur_states.sort_values(by='timestamp')
    cur_states.reset_index(drop=True, inplace=True)
    cur_states['state_vector'] = cur_states['state_vector'].apply(lambda s: eval(s)[3:5] + [eval(s)[8]])
    
    ref_states = pd.read_csv(ref_state_file)
    ref_states['timestamp'] = ref_states['time'].apply(lambda t: datetime.strptime(t[:-3], "%Y/%m/%d %H:%M:%S.%f"))
    
    return cur_states, ref_states

def match_cur_ref(cur_states, ref_states):

    ref_index = 0
    ref_timestamp = ref_states['timestamp'][0]
    refs = []
    ref_size = len(ref_states)

    for cur_index, row in cur_states.iterrows():
        if ref_index + 1>= ref_size:
            refs.append(ref_states[['vn', 've', 'yaw']].iloc[ref_index].tolist())
        elif row['timestamp'] > ref_states['timestamp'][ref_index+1]:
            ref_index += 1
            refs.append(ref_states[['vn', 've', 'yaw']].iloc[ref_index].tolist())
        elif row['timestamp'] > ref_states['timestamp'][ref_index]:
            refs.append(ref_states[['vn', 've', 'yaw']].iloc[ref_index].tolist())
        else:
            # Error msg
            print(cur_index, row['timestamp'], ref_states['timestamp'][ref_index])
            refs.append(np.nan)
    cur_states['ref_state'] = refs
    return

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=54):
    size = len(X)
    train_num = int(size*train_ratio)
    val_num = int(size*val_ratio)
    
    # randomly shuffling X and y in unison
    # np.random.seed(random_seed)
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    X_train, X_val, X_test = X[:train_num], X[train_num:train_num+val_num], X[train_num+val_num:]
    y_train, y_val, y_test = y[:train_num], y[train_num:train_num+val_num], y[train_num+val_num:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_past_state_X(X, n_visible=5):
    X_ps = []
    temp = X[:n_visible].reshape(-1)
    for i in range(n_visible, len(X)):
        X_ps.append(temp)
        temp = np.append((temp[3:]), X[i])
    return np.stack(X_ps)


if __name__ == "__main__":

    cur_state_file = "../first_collection/current_state.csv"
    ref_state_file = "../first_collection/reference_state.csv"

    cur_states, ref_states = read_csv(cur_state_file, ref_state_file)
    match_cur_ref(cur_states, ref_states)

    cur_states_dropna = cur_states.dropna()
    cur_states_list = cur_states_dropna["state_vector"].tolist()
    ref_states_list = cur_states_dropna["ref_state"].tolist()

    with open("cur_states.npy", "wb") as f:
        np.save(f, np.array(cur_states_list))
    with open("ref_states.npy", "wb") as f:
        np.save(f, np.array(ref_states_list))
    # cur_states.to_csv("../first_collection/processed_states.csv")

    # states = pd.read_csv("../first_collection/processed_states.csv")
    # print(states["state_vector"][0][0])