"""
Adapted from https://github.com/rampopat/uncertainty/blob/main/Main_Notebook_DementiaBank.ipynb

"""

from scipy.stats import entropy
import numpy as np
import baal

def binarize(ps, threshold=0.5):
    return np.where(ps > threshold, 1, 0)

def get_entropy(ps):
    # ps is df series of the preds 
    # entropy(p) = -((1-p)log_2(1-p) + plog_2(p)) for p scalar,
    # we map this over the vector of probabilities
    # NOTE: WE CHANGED TO BASE E, TO ALLOW COMPARISON WITH BATCHBALD
    # IN PRINCIPLE THE BASE DOESN'T CHANGE THE RANKINGS
    return entropy(pk=np.stack([1-ps, ps]), axis=0)

def get_predictive_entropy(all_preds):
    # this is for convenience, so we can call all uncertainty metrics uniformly
    # all_preds.shape = (M, data_size)
    return get_entropy(all_preds.mean(axis=0))

def get_expected_entropy(all_preds):
    # all_preds.shape = (M = ensemble_size, data_size)
    ensemble_size = all_preds.shape[0]
    individual_entrops = np.array([get_entropy(all_preds[m]) for m in range(ensemble_size)])
    return individual_entrops.mean(axis=0)

def get_mutual_info(all_preds):
    # all_preds.shape = (M, data_size)
    # predictive_entropy.shape = (data_size,)
    predictive_entropy = get_entropy(all_preds.mean(axis=0))
    expected_entropy = get_expected_entropy(all_preds)
    return predictive_entropy - expected_entropy

def get_variation_ratio(all_preds, threshold=0.5):
    # all_preds.shape = (M, data_size)
    # ps, mean_preds, vrs shape = (data_size)
    data_size = all_preds.shape[1]
    ps = binarize(all_preds, threshold=threshold).mean(axis=0) # probability of dementia
    mean_preds = binarize(all_preds.mean(axis=0), threshold=threshold)
    vrs = [ps[i] if (mean_preds[i] == 0) else (1 - ps[i]) for i in range(data_size)]
    return vrs

def get_pred_var(all_preds):
    return all_preds.var(axis=0)

def get_batch_bald(all_preds, ndata_to_label, num_draw=5000):
    # this returns ranks not the uncertainty scores
    ps = all_preds.T # [ensemble_size, pool_size]  
    qs = np.stack([1-ps, ps], axis=1)
    return np.flip(baal.active.heuristics.BatchBALD(num_samples=ndata_to_label, num_draw=num_draw)(qs))[:ndata_to_label]