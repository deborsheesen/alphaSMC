# Calling libraries:
from __future__ import division
get_ipython().magic('matplotlib inline')
import numpy as np, numpy.random as npr, scipy, copy, networkx as nx

def initialise(data, theta, propagate, N) :
    
    x_0, y = data['x_0'], data['y']
    hid_dim = len(x_0)
    
    T, obs_dim = np.shape(y)
    particles = np.zeros((N,hid_dim))
    particles[:] = x_0
    weights = np.ones(N)
    log_NC = np.zeros(T+1)
    particles = propagate(particles, theta)
    test_fn_est = np.zeros(T)
    
    W = np.ones(N)
    w = np.ones(N)
    resampled_idx = [1]*N
    
    return particles, weights, log_NC, test_fn_est, W, resampled_idx


# Bootstrap particle filter:

def bootstrap_PF(data, theta, potential, propagate, test_fn, N) :
    
    y = data['y']
    particles, _, log_NC, test_fn_est, _, _ = initialise(data, theta, propagate, N)
    T = np.shape(y)[0]
    
    for t in range(T) :
        weights = potential(particles, y[t], theta)/N
        log_NC[t+1] = log_NC[t] + np.log(np.sum(weights))
        particles = particles[npr.choice(a=N,size=N,replace=True,p=weights/np.sum(weights))]
        particles = propagate(particles, theta)
        test_fn_est[t] = np.mean(test_fn(particles))
    return log_NC, test_fn_est


# alphaSMC with random connections:
def alphaSMC_random(data, theta, potential, propagate, test_fn, N, C) :
    
    y = data['y']
    particles, weights, log_NC, test_fn_est, W, resampled_idx = initialise(data, theta, propagate, N)
    T = np.shape(y)[0]
    
    for t in range(T) :
        weights *= potential(particles, y[t], theta)
        for particle in range(N) :
            connections = npr.choice(a=N, size=C, replace=False)
            W[particle] = np.mean(weights[connections])
            resampled_idx[particle] = npr.choice(a=connections, p=weights[connections]/np.sum(weights[connections]))
        particles = particles[resampled_idx]
        weights[:] = W[:]
        log_NC[t+1] = np.log(np.mean(weights))
        particles = propagate(particles, theta)
        test_fn_est[t] = np.sum(weights*test_fn(particles))/np.sum(weights)
    
    return log_NC, test_fn_est


def vectorized(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = npr.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]

def alphaSMC(data, theta, potential, propagate, test_fn, N, C, alpha) :
    
    if type(alpha) == np.ndarray : 
        assert N == np.shape(alpha)[0]
    
    y = data['y']
    
    particles, weights, log_NC, test_fn_est, _, _ = initialise(data, theta, propagate, N)
    T = np.shape(y)[0]
    prob_wts = np.ones((N,N))
    items = np.arange(N)
    weights = weights/N

    for t in range(T) :
        if type(alpha) == np.ndarray :
            alpha_matrix = alpha
        else :
            alpha_matrix = d_regular_graph(N, C, fix_seed=True)
        prob_wts = alpha_matrix*weights*potential(particles, y[t], theta) 
        W_bar = np.sum(prob_wts,axis=1)
        prob_matrix = (prob_wts/np.sum(prob_wts, axis=1, keepdims=True)).T
        resampled_particles = vectorized(prob_matrix, items)
        particles[:] = particles[resampled_particles]
        log_NC[t+1] = log_NC[t] + np.log(np.sum(W_bar)) 
        weights = W_bar/np.sum(W_bar)         
        particles = propagate(particles, theta)
        test_fn_est[t] = np.sum(W_bar*test_fn(particles))/np.sum(W_bar)
    
    return log_NC, test_fn_est


def d_regular_graph(N, C, fix_seed=True) :
    if fix_seed == True :
        G = nx.random_regular_graph(C, N, 12345)
    else :
        G = nx.random_regular_graph(C, N)
    A = np.asarray(nx.to_scipy_sparse_matrix(G).todense())
    A = A/np.sum(A,1)
    return A

# Local exchange matrix:
def local_exchange_graph(N, C) :
    A = np.diag(np.ones(N))/C
    for i in range(N) :
        d, d_left, d_right = 1, 1, 1
        while d < C : 
            A[i, (i+1+d_right)%N-1] = 1/C
            d_right += 1; d += 1
            if d < C :
                A[i, (i+1-d_left)%N-1] = 1/C
                d_left += 1; d += 1
    return A


