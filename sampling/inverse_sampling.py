import numpy as np
import matplotlib.pyplot as plt


def inverse_sample(cur_samples, num_bins=100, max_val=1, plot=False):
    bins = np.linspace(-max_val, max_val, num_bins)
    counts, bins, bars = plt.hist(cur_samples, bins)
    plt.xlabel("Velocity")
    plt.ylabel("Frequency")
    # print(counts, bins)
    
    size = len(cur_samples)
    prob_dist_cur = counts / size
    prob_dist_inv = (1 / prob_dist_cur) / sum(1/prob_dist_cur)
    plt.figure()
    plt.bar(bins[:-1], prob_dist_inv, width=max_val*2/(num_bins-1))
    plt.xlabel("Velocity")
    plt.ylabel("Probability")
    if plot:
        plt.show()
    return bins[:-1], prob_dist_inv

# TODO: Finish this method
def inverse_sample_nd(cur_samples, num_bins, dim=2, max_val=1, plot=False):
    return

if __name__ == "__main__":
    cur_states = np.load("../second_collection_slower/cur_states.npy")
    ref_states = np.load("../second_collection_slower/ref_states.npy")
    bins, prob_dist_inv = inverse_sample(cur_states[:,1], plot=False)
    bin_size = bins[1] - bins[0]

    # Here's an example of sampling based on this new inverse distribution
    assert len(bins) == len(prob_dist_inv)
    example_ind = np.random.choice(range(len(prob_dist_inv)), p=prob_dist_inv)
    example = bins[example_ind] + np.random.random() * bin_size
    print(example)
    plt.figure()
    plt.hist(ref_states[:,0], bins=np.linspace(-1, 1, 20))
    plt.xlabel("Reference velocity")
    plt.ylabel("Frequency")
    plt.show()