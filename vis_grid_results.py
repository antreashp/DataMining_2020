
import skopt.plots
from train_grid import *
from skopt import dump, load
import matplotlib.pyplot as plt
# import pickle
def main():
    # filename = 'results_test.pickle'
    # with open(filename, 'rb') as f:
    results = load('result.gz')
    print(results)
    skopt.plots.plot_convergence(results)
    plt.show()
if __name__ == "__main__":
    main()
    