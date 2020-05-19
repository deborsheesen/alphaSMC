from __future__ import division
import numpy as np, scipy, numpy.random as npr
from scipy.stats import norm

def propagate_SV(particles, theta) :
    return theta[0]*particles + theta[1]*npr.randn(*np.shape(particles))

def potential_SV(y, X, theta) :
    return (norm.pdf(x=y, loc=0, scale=theta[2]*np.exp(X/2))).flatten()
 
def log_potential_SV(y, X, theta) :
    return (norm.logpdf(x=y, loc=0, scale=theta[2]*np.exp(X/2))).flatten()

def simulate_data_SV(theta, x_0, T) :
    dim = len(x_0)
    X = np.zeros((T+1,dim))
    X[0] = x_0
    y = np.zeros((T, dim))
    for t in range(T) :
        X[t+1] = propagate_SV(X[t], theta)
        y[t] = norm.rvs(loc=0, scale=theta[2]*np.exp(X[t+1]/2))
        # y[t] = theta[2]*np.exp(X[t+1]/2)*npr.randn(dim)
    return y, X
