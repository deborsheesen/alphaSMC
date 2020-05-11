from __future__ import division
import numpy as np, numpy.random as npr
from scipy.stats import norm

def propagate_Lorenz63(particles, theta) :
    """
    usual: 3D Lorenz model + multiplicative noise
    compute one step forward, usual Euler-Maruyama discretization
    """
    sigma, rho, beta, noise_intensity, dt, delta = theta[:6]
    #ODE forward + noise
    sqdt = np.sqrt(dt)
    state = np.zeros(np.shape(particles))
    for i in range(int(delta/dt)) :
        x, y, z = particles[:,0], particles[:,1], particles[:,2]
        W = npr.randn(*np.shape(particles))
        state[:,0] = x + dt*sigma*(y - x) #+ noise_intensity*x*sqdt*W[0]
        state[:,1] = y + dt*(x*(rho - z) - y) #+ noise_intensity*y*sqdt*W[1]
        state[:,2] = z + dt*(x*y - beta*z) #+ noise_intensity*z*sqdt*W[2]
        particles = state + noise_intensity*particles*sqdt*W
    return particles

def f(x) :
    return x
    #return 1/(1+np.exp(-x))

def log_potential_Lorenz63(y, X, theta) :
    obs_noise = theta[6]
    return np.sum(norm.logpdf(x=y, loc=f(X), scale=obs_noise),1)

def potential_Lorenz63(y, X, theta) :
    obs_noise = theta[6]
    return np.prod(norm.pdf(x=y, loc=f(X), scale=obs_noise),1)
    #return np.prod(np.exp(-np.abs(X))*y**np.abs(X),1)

def simulate_data_Lorenz63(theta, x_0, T) : 
    """
    generate a sequence of observations | x_0 : initialization
    T = number of observations | delta = time between observations
    """
    y = np.zeros((T,3))
    X = np.zeros((T+1,3))
    X[0] = x_0
    obs_noise = theta[6]
    for t in range(T) :
        X[t+1] = propagate_Lorenz63(np.reshape(X[t],[1,3]), theta)
        #y[t] = npr.poisson(lam=np.abs(X[t+1]))
        y[t] = f(X[t+1]) + obs_noise*npr.randn(3)
    return y, X




                                  
                                  
                                  
                                  
