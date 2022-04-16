import matplotlib.pyplot as plt
import numpy as np

def plot_dist(states):
    # Plot the distribution of states traversed during the data collection of the robot
    bins = np.linspace(-1.5, 1.5, 20)
    plt.hist(states[:,1], bins=bins)
    plt.show()

if __name__ == "__main__":
    
    cur_states = np.load("../first_collection/cur_states.npy")
    ref_states = np.load("../first_collection/ref_states.npy")
    plot_dist(ref_states)
