import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    h2g_hess = np.loadtxt("h2g_hess.txt", delimiter=",")
    plt.hist(h2g_hess, bins=30)
    plt.show()