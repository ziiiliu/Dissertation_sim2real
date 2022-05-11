"""
This file implements the possible metrics used to evaluate the performance of the models
"""

from pyexpat import model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw
import pandas as pd
import os

def avg_distance_from_start(real_coords, pred_coords):
    """
    calculates the average displacement between the real robot and modelled robot
    from start to end. Does not reflect how error has propagated
    ---------------
    Params
    ---------------
    real_coords: list
        2D list of (x, y) coordinates.
    """
    real_coords, pred_coords = np.asarray(real_coords), np.asarray(pred_coords)
    all_dist = np.linalg.norm(real_coords-pred_coords,axis=1)
    dist_avg, dist_std = np.mean(all_dist), np.std(all_dist)
    return dist_avg, dist_std

    
def propagated_error(real_coords, pred_coords, n=5):
    """
    calculates a score that describes how quickly the modelled robot deviates from the real robot
    """
    real_coords, pred_coords = np.asarray(real_coords), np.asarray(pred_coords)
    all_dist = np.linalg.norm(real_coords-pred_coords,axis=1)
    all_abs_diff = abs(np.diff(all_dist, n=n))
    percentage_changes = all_abs_diff[1:] / all_dist[1:-n]
    print(percentage_changes)
    avg_change = np.mean(percentage_changes)
    return avg_change
    

def lag(real_coords, pred_coords):
    """
    calculates a score that describes the lag of the modelled robot behind the real robot
    """

    pass

def parallel_degree_step(real_vel, pred_vel):
    """
    describes the angle between two trajectories
    """
    if np.isnan(pred_vel):
        return 0
    return cosine_similarity(real_vel, pred_vel)

def parallel_degree_avg(real_coords, pred_coords):
    real_dis, pred_dis = np.diff(real_coords, axis=0), np.diff(pred_coords, axis=0)
    if np.isnan(np.sum(pred_dis)) or np.isnan(np.sum(real_dis)):
        return np.nan, np.nan
    all_sim = cosine_similarity(real_dis, pred_dis)
    all_sim = np.diagonal(all_sim)
    return np.mean(all_sim), np.std(all_sim)

# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
def dtw_distance(real_coords, pred_coords):
    x_distance = dtw.distance(real_coords[:, 0], pred_coords[:, 0])
    y_distance = dtw.distance(real_coords[:, 1], pred_coords[:, 1])
    return np.sqrt(x_distance**2 + y_distance**2)
# TODO
# ref: https://arxiv.org/pdf/2110.03267.pdf
def delta_empirical_sigma_value():
    pass

if __name__ == "__main__":
    print(os.getcwd())
    dir = "../trajectories/"
    traj_name = dir + "simplepredictor_5000_steps_from_start"
    real_coords = (np.load(traj_name+"/real.npy")).astype(np.double)
    modelled_coords = (np.load(traj_name+"/modelled.npy")).astype(np.double)
    ideal_coords = (np.load(traj_name+"/ideal.npy")).astype(np.double)

    # Average distance from start
    real_modelled_avg_dis = avg_distance_from_start(real_coords, modelled_coords)
    real_ideal_avg_dis = avg_distance_from_start(real_coords, ideal_coords)
    modelled_ideal_avg_dis = avg_distance_from_start(modelled_coords, ideal_coords)

    # Propagated Error
    real_modelled_pe = propagated_error(real_coords, modelled_coords)
    real_ideal_pe = propagated_error(real_coords, ideal_coords)
    modelled_ideal_pe = propagated_error(modelled_coords, ideal_coords)

    # Parallel degree avg
    real_modelled_pda, _ = parallel_degree_avg(real_coords, modelled_coords)
    real_ideal_pda, _ = parallel_degree_avg(real_coords, ideal_coords)
    modelled_ideal_pda, _ = parallel_degree_avg(modelled_coords, ideal_coords)

    # dtw
    real_modelled_dtw = dtw_distance(real_coords, modelled_coords)
    real_ideal_dtw = dtw_distance(real_coords, ideal_coords)
    modelled_ideal_dtw = dtw_distance(modelled_coords, ideal_coords)

    # collecting evaluation results
    real_modelled_metrics = [real_modelled_avg_dis, real_modelled_pe, real_modelled_pda, real_modelled_dtw]
    real_ideal_metrics = [real_ideal_avg_dis, real_ideal_pe, real_ideal_pda, real_ideal_dtw]
    modelled_ideal_metrics = [modelled_ideal_avg_dis, modelled_ideal_pe, modelled_ideal_pda, modelled_ideal_dtw]

    metrics = pd.DataFrame(data={ "real-modelled": real_modelled_metrics,
                        "real-ideal": real_ideal_metrics,
                        "modelled-ideal": modelled_ideal_metrics}, index=["avg_dis", "pe", "pda", "dtw"])
    
    metrics.to_csv(traj_name+"/metrics.csv")

