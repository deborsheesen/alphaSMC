from __future__ import division
import numpy as np, scipy, numpy.random as npr
from scipy.stats import norm

def simulate_data_SV(theta, x_0, T) :
    dim = len(x_0)
    X = x_0
    y = np.zeros((T, dim))
    for t in range(T) :
        X = theta[0]*X + theta[1]*npr.randn(dim)
        y[t] = theta[2]*np.exp(X/2)*npr.randn(dim)
    return y

def potential_SV(y, X, theta) :
    return (norm.pdf(x=y, loc=0, scale=theta[2]*np.exp(X/2))).flatten()

def propagate_SV(particles, theta) :
    return theta[0]*particles + theta[1]*npr.randn(*np.shape(particles))