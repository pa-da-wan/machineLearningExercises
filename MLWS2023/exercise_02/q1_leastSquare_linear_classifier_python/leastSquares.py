import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    ones = np.ones(data.shape[0]).reshape(data.shape[0],1)
    X = np.hstack((ones, data))
    Y = label
    W = np.linalg.inv(X.T@X)@X.T@Y
    weight = W[1:]
    bias = W[0]
    return weight, bias
