
# coding: utf-8

# In[7]:

# Calling libraries:
from __future__ import division
get_ipython().magic('matplotlib inline')
import numpy as np, time, math, scipy, networkx as nx,  matplotlib.pyplot as plt
from scipy.stats import norm, uniform, bernoulli
from scipy.linalg import sqrtm 
from pylab import plot, show, legend
plt.rcParams['figure.figsize'] = (15.0, 3.0)
int = np.vectorize(int)
from tqdm import trange

def simulate_data(theta,x_0,T,delta) :
    X = x_0
    y = np.zeros(T)
    for t in range(T) :
        X += -theta[0]*(X-theta[1])*delta + np.sqrt(delta)*theta[2]*np.random.randn(1)
        y[t] = X + theta[3]*np.random.randn(1) 
    return y



def potential(y,X,sigma_error) :
    return np.exp(-1/(2*sigma_error**2)*(X-y)**2)


# Bootstrap particle filter:
# 
# Resample at every time step.


def bootstrap_PF(n_particles, theta, x_0, y, delta, potential, test_function ) :
    
    scipy.random.seed()
    
    T = len(y)
    particles    = np.zeros(n_particles)
    particles[:] = x_0
    log_NC       = np.zeros(T+1)
    particles   += -theta[0]*(particles-theta[1])*delta + np.sqrt(delta)*theta[2]*np.random.randn(n_particles)
    test_function_estimate = np.zeros(T)

    for t in range(T) :
        weights     = potential( particles, y[t], theta[3] ) / n_particles
        log_NC[t+1] = log_NC[t] + np.log(np.sum(weights))
        particles   = particles[np.random.choice(a=n_particles,size=n_particles,replace=True,p=weights/np.sum(weights))]
        particles  += -theta[0]*(particles-theta[1])*delta + np.sqrt(delta)*theta[2]*np.random.randn(n_particles)
        
        test_function_estimate[t] = np.mean(test_function(particles))
        
    return log_NC[1::], test_function_estimate


# $\alpha$SMC with random connections:
# 
# Choose a random $\alpha$ matrix at each time.


def alphaSMC_random(n_particles, theta, x_0, y, delta, n_connections, potential, test_function) :
    
    scipy.random.seed()
    
    T = len(y)
    particles    = np.zeros(n_particles)
    particles[:] = x_0
    log_NC       = np.zeros(T+1)
    particles   += -theta[0]*(particles-theta[1])*delta + np.sqrt(delta)*theta[2]*np.random.randn(n_particles)
    test_function_estimate = np.zeros(T)
    
    W = np.ones(n_particles)
    weights = np.ones(n_particles) 
    resampled_index = [1]*n_particles

    for t in range(T) :
        weights *= potential(particles, y[t], theta[3])
        for particle in range(n_particles) : 
            connections = np.random.choice(n_particles, n_connections, False)
            W[particle] = np.sum(weights[connections]) / n_connections
            resampled_index[particle] = np.random.choice(a=connections,
                                                         p=weights[connections]/np.sum(weights[connections]))
        particles[:] = particles[resampled_index]
        log_NC [t+1] = log_NC[t] + np.log(np.sum(W)) 
        weights  [:] = W / np.sum(W) 
        particles += -theta[0]*(particles-theta[1])*delta + np.sqrt(delta)*theta[2]*np.random.randn(n_particles)
        
        test_function_estimate[t] = np.sum(weights*test_function(particles))
    
    return log_NC[1:], test_function_estimate


# $\alpha$SMC with fixed $\alpha$ matrix:

def vectorized(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]




def alphaSMC_fixed_alpha(alpha_matrix, theta, x_0, y, delta, potential, test_function) :
    
    scipy.random.seed()
    
    T = len(y)
    n_particles     = np.shape(alpha_matrix)[0]
    particles       = np.zeros(n_particles)
    particles[:]    = x_0
    weights         = np.ones(n_particles)
    log_NC          = np.zeros(T+1)
    particles += -theta[0]*(particles-theta[1])*delta + np.sqrt(delta)*theta[2]*np.random.randn(n_particles)
    test_function_estimate = np.zeros(T)
    
    prob_wts = np.ones(( n_particles, n_particles ))
    items = np.arange(n_particles)

    for t in range(T) :
        prob_wts[:] = alpha_matrix*weights*potential(particles, y[t], theta[3]) 
        W_bar = np.sum(prob_wts,axis=1)
        prob_matrix = (prob_wts / np.sum(prob_wts, axis=1, keepdims=True)).T
        resampled_particles = vectorized(prob_matrix, items)
        particles[:] = particles[resampled_particles]
        
        log_NC[t+1] = log_NC[t] + np.log(np.sum(W_bar)) 
        weights = W_bar / np.sum(W_bar)         
        particles += -theta[0]*(particles-theta[1])*delta + np.sqrt(delta)*theta[2]*np.random.randn(n_particles)
        
        test_function_estimate[t] = np.sum(weights/np.sum(weights)*test_function(particles))
    
    return log_NC[1:], test_function_estimate

def d_regular_graph(n_particles,n_connections,fix_seed=True) :
    if fix_seed == True :
        G = nx.random_regular_graph(n_connections, n_particles, 12345)
    else :
        G = nx.random_regular_graph(n_connections, n_particles)
    A = np.asarray(nx.to_scipy_sparse_matrix(G).todense())
    A = A / A.sum(axis=1)[0]
    return A, G


# Local exchange matrix:


def local_exchange_graph(n_computers, degree) :
    A = np.diag(np.ones(n_computers))/degree
    for i in range(n_computers) :
        d, d_left, d_right = 1, 1, 1
        while d < degree : 
            A[i, (i+1+d_right)%n_computers-1] = 1/degree
            d_right += 1; d += 1
            if d < degree :
                A[i, (i+1-d_left)%n_computers-1] = 1/degree
                d_left += 1; d += 1
    return A





