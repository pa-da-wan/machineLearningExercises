import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created
    N = len(samples)
    pos =  np.arange(-5,5.0,0.1).reshape(-1,1)
    prob = np.zeros(pos.shape)

    dist = np.abs((pos-samples.reshape(-1,N))) #100xN
    sorted_dist = np.argsort(dist, axis = -1)   #100xN
    v= dist[np.where(sorted_dist ==k-1)].reshape(-1,1)       #100x1
    prob = k/(2*N*v).reshape(-1,1)            #100x1
    estDensity  = np.hstack((pos,prob))

    return estDensity
