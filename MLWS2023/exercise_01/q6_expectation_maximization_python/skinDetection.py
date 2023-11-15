import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####
    height, width = img.shape[0], img.shape[1]
    skin = np.zeros((height,width))
    noSkin = np.zeros((height,width))


    s_weights, s_means, s_covariances= estGaussMixEM(sdata, K, n_iter, epsilon)
   

    n_weights, n_means, n_covariances= estGaussMixEM(ndata, K, n_iter, epsilon)
    

    for i in range(height):
        for j in range(width):
            skin[i,j] = getLogLikelihood(s_means, s_weights, s_covariances, img[i,j])
            noSkin[i,j] = getLogLikelihood(n_means, n_weights, n_covariances, img[i,j])
            

    likelihood_ratio = np.exp(skin-noSkin)
    result = np.where(likelihood_ratio > theta,1,0) 

    return result
