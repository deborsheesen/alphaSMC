from __future__ import division
import numpy as np, scipy, numpy.random as npr
from scipy.stats import norm

def simulate_data_OU(theta, x_0, T) :
    X = x_0
    dim = len(x_0)
    y = np.zeros((T,dim))
    for t in range(T) :
        X = X - theta[0]*(X-theta[1])*theta[4] + np.sqrt(theta[4])*theta[2]*npr.randn(dim)
        y[t] = X + theta[3]*npr.randn(dim) 
    return y

def potential_OU(y, X, theta) :
    return (np.exp(-1/(2*theta[3]**2)*(X-y)**2)).flatten()

def propagate_OU(particles, theta) :
    return particles - theta[0]*(particles-theta[1])*theta[4] + np.sqrt(theta[4])*theta[2]*npr.randn(*np.shape(particles))