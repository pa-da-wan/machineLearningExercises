import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    N = len(samples)
    pos = np.arange(-5,5.0,0.1).reshape((-1,1))
    kernel = np.exp(-0.5*((pos-samples.reshape((-1,N)))/(h))**2)/np.sqrt(2*np.pi*h**2)
    prob = np.sum(kernel, axis =1, keepdims=True)/N
    estDensity = np.hstack((pos, prob))

    return estDensity
