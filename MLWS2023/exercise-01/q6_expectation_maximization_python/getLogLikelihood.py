import numpy as np
def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####

    K,D = np.array(means).shape
    N = X.shape[0]
    
    logLikelihood=0
    for n in range(N):
        weighted_gauss = 0
        for i in range(K):
            exponent = (X[n]-means[i])@np.linalg.inv(covariances[:,:,i])@(X[n]-means[i]).T
            base = (((2*np.pi)**(D/2))*np.sqrt(np.linalg.det(covariances[:,:,i])))
            weight = np.array(weights).reshape((1,-1))[0,i]
            weighted_gauss+= weight*np.exp(-0.5*exponent)/base
        logLikelihood+= np.log(weighted_gauss)
    return logLikelihood

