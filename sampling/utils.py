import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score, explained_variance_score, mean_absolute_error


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=54):
    size = len(X)
    train_num = int(size*train_ratio)
    val_num = int(size*val_ratio)
    
    # randomly shuffling X and y in unison
    np.random.seed(random_seed)
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    X_train, X_val, X_test = X[:train_num], X[train_num:train_num+val_num], X[train_num+val_num:]
    y_train, y_val, y_test = y[:train_num], y[train_num:train_num+val_num], y[train_num+val_num:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_performance(metric, preds, labels):
    return metric(labels, preds)
    
def al_performance(active_test_preds, test_labels, train_sizes, best_epoch_performances=None, title='Active Learning', plotting=False):
    # for ensembles, active_test_preds is the mean across the ensemble, so we can use it for single model
    # active test preds should be un-rinted i.e. in (0, 1)
    PMETRIC_FNS = [mean_squared_error, r2_score, explained_variance_score] # precision_score, recall_score]
    PMETRIC_NAMES = ['MSE', 'R2', 'EVS'] #'Precision', 'Recall']
    al_steps = active_test_preds.shape[0]
    scores = np.zeros((len(PMETRIC_FNS), al_steps))
    for al_step in range(al_steps):
        preds = active_test_preds[al_step, :]
        for i, pmetric in enumerate(PMETRIC_FNS):
            scores[i, al_step] = get_performance(pmetric, preds, test_labels)

    if plotting:
        for i, name in enumerate(PMETRIC_NAMES):        
            plt.plot(train_sizes, scores[i, :], marker='.', label=name)
        # if best_epoch_performances is not None:
        #     plt.plot(train_sizes, best_epoch_performances, marker='.', label='ES Accuracy')
        plt.xlabel('Train Set Size')
        plt.ylabel('Performance')
        plt.title(title)
        plt.legend()
        plt.show()
    return scores

def compare_performances(al_list_preds, labels, train_sizes, baseline_preds=None, strategy_names=None, mask=None, 
                         is_ensemble=True):
    # make sure al_list_preds = input a list even if there is only one al_strategy
    PMETRIC_FNS = [mean_squared_error, r2_score, explained_variance_score] # precision_score, recall_score]
    PMETRIC_NAMES = ['MSE', 'R2', 'EVS'] #'Precision', 'Recall']
    # baseline_preds is a non-AL-based deep ensemble baseline that we will draw as a point on the graph
    # baseline_preds.shape = (ensemble_size, test_size)
    if baseline_preds is not None:
        baseline_preds = baseline_preds.mean(axis=0)
        baseline_scores = [get_performance(metric, baseline_preds, labels) for metric in PMETRIC_FNS]
    # mask is the subset of the data that we want
    if isinstance(al_list_preds, list):
        al_list_preds = np.array(al_list_preds)
    # al_list_preds is np array of ensembled preds by different strategies
    # al_list_preds.shape = (al_strategies, al_steps, ensemble_size, test_size)
    if strategy_names is None:
        strategy_names = [' '] * len(al_list_preds)
    if train_sizes is None:
        al_steps = al_list_preds.shape[1]
        train_sizes = np.linspace(0, 1, num=al_steps)
    # mean is over the ensemble_size because preds.shape = al_list_preds.shape[1:]
    if is_ensemble: # we set is_ensemble = False 
        al_list_preds = [preds.mean(axis=1) for preds in al_list_preds]
    list_scores = [al_performance(preds, labels, train_sizes) for preds in al_list_preds]
    print(list_scores, train_sizes)
    rows = 1
    cols = len(PMETRIC_NAMES)
    plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    for i, name in enumerate(PMETRIC_NAMES):        
        plt.subplot(rows, cols, i + 1)
        for j, scores in enumerate(list_scores):
            if mask is None or j in mask:
                plt.plot(train_sizes, list_scores[j][i, :], marker='.', label=strategy_names[j])
        if baseline_preds is not None:
            plt.scatter(train_sizes[-1], baseline_scores[i], marker='^', label='Non-AL Baseline')
        plt.xlabel('Train Set Size')        
        plt.ylabel('Performance')
        plt.title(name)
        plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('al-comparison', dpi=350)
    plt.show()