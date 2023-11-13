import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    N,K = gamma.shape
    # X = np.array(X)
    # gamma = np.array(gamma)
    D = X.shape[1]
    N_j=gamma.sum(axis = 0, keepdims = True)
    weights = np.squeeze(N_j/N)

    means = (gamma.T @ X)/(N_j.T )  #shape (KxD)
    covariances = np.zeros((D,D,K))

    for i in range(K):
        covariances[:, :, i] = np.dot((gamma[:, i] * (X - means[i]).T), (X - means[i])) / N_j[0, i]


    logLikelihood = getLogLikelihood(means= means, weights= weights, covariances= covariances, X=X)
    return weights, means, covariances, logLikelihood
