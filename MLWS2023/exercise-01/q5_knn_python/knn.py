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
    N= len(samples)
    pos = np.arange(-5, 5, 0.1).reshape(-1,1)
    distance = np.sort(np.abs(pos - samples.reshape(-1,N)), axis =1)
    max_radius = distance[:,k-1].reshape(N,-1)
    prob = k/(N*2*max_radius)
    estDensity = np.hstack((pos,prob))


    return estDensity
