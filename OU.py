from __future__ import division
import numpy as np, scipy, numpy.random as npr
from scipy.stats import norm


def propagate_OU(particles, theta) :
    return particles - theta[0]*(particles-theta[1])*theta[4] + np.sqrt(theta[4])*theta[2]*npr.randn(*np.shape(particles))

def log_potential_OU(y, X, theta) :
    return (norm.logpdf(x=y, loc=X, scale=theta[3])).flatten()
    
def potential_OU(y, X, theta) :
    return (norm.pdf(x=y, loc=X, scale=theta[3])).flatten()

def simulate_data_OU(theta, x_0, T) :
    dim = len(x_0)
    X = np.zeros((T+1,dim))
    X[0] = x_0
    y = np.zeros((T, dim))
    for t in range(T) :
        X[t+1] = propagate_OU(X[t], theta)
        y[t] = X[t+1] + theta[3]*npr.randn(dim) 
    return y, X