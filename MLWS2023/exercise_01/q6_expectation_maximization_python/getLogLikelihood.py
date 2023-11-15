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


    K = np.array(weights).reshape((1,-1)).shape[1]

    if len(X.shape) > 1:
        N,D = np.array(X).shape
    else:
        N = 1
        D = X.shape[0]
    
    logLikelihood=0
    for n in range(N):
        weighted_gauss = 0
        for i in range(K):

            if N == 1:
                meansDiff = X - means[i]
            else:
                meansDiff = X[n,:] - means[i]
            
            covariance = covariances[:, :, i].copy()
            numerator = np.exp(-0.5*(meansDiff)@np.linalg.inv(covariance)@(meansDiff).T)
            norm = 1. / float(((2 * np.pi) ** (float(D) / 2.)) * np.sqrt(np.linalg.det(covariance)))

            weight = np.array(weights).reshape((1,-1))[0,i]
            
            weighted_gauss+= weight*numerator*norm
        logLikelihood+= np.log(weighted_gauss)
    return logLikelihood

