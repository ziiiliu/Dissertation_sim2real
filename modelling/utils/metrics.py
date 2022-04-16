"""
This file implements the possible metrics used to evaluate the performance of the models
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    percentage_changes = all_abs_diff / all_dist[:n]
    print(percentage_changes)
    avg_change = np.mean(percentage_changes)
    return avg_change
    

# TODO
def lag(real_coords, pred_coords):
    """
    calculates a score that describes the lag of the modelled robot behind the real robot
    """

    pass

def parallel_degree_step(real_vel, pred_vel):
    """
    describes the angle between two trajectories
    """
    return cosine_similarity(real_vel, pred_vel)

def parallel_degree_avg(real_coords, pred_coords):
    real_dis, pred_dis = np.diff(real_coords, axis=0), np.diff(pred_coords, axis=0)
    all_sim = cosine_similarity(real_dis, pred_dis)
    return np.mean(all_sim), np.std(all_sim)

# TODO
def dtw_distance():
    pass

# TODO
# ref: https://arxiv.org/pdf/2110.03267.pdf
def delta_empirical_sigma_value():
    pass
