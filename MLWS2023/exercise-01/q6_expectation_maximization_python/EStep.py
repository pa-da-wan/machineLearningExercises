import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    logLikelihood = getLogLikelihood(means=means, weights=weights, covariances=covariances, X=X)
    K, D = np.array(means).shape
    N = X.shape[0]

    gamma = np.zeros((N, K))

    for i in range(K):
        diff = X - means[i]
        inv_cov = np.linalg.pinv(covariances[:, :, i])
        weight = np.array(weights).reshape((1,-1))[0,i]
        num = weight * np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))
        denom = np.sqrt(np.linalg.det(covariances[:, :, i]) * (2 * np.pi) ** D)
        gamma[:, i] = num / denom

    # Normalize gamma along axis 1
    gamma /= gamma.sum(axis=1, keepdims=True)
    return [logLikelihood, gamma]
