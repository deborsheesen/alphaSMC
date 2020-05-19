# Calling libraries:
from __future__ import division
get_ipython().magic('matplotlib inline')
import numpy as np, numpy.random as npr, scipy, copy, networkx as nx
from tqdm import trange

################################################################################################################
############################################### INITIALISE STUFF ###############################################
################################################################################################################

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


################################################################################################################
########################################## BOOTSTRAP PARTICLE FILTER ###########################################
################################################################################################################

# def bootstrap_PF(data, theta, potential, propagate, test_fn, N) :
#    
#     y = data['y']
#     particles, _, log_NC, test_fn_est, _, _ = initialise(data, theta, propagate, N)
#     T = np.shape(y)[0]
#    
#     for t in trange(T) :
#         #log_weights = log_potential(particles, y[t], theta)
#         #weights = np.exp(log_weights-min(log_weights))
#         #log_NC_increment = np.log(np.mean(np.log(weights)) + min(log_weights))
#         weights = potential(particles, y[t], theta)/N
#         log_NC_increment = np.log(np.sum(weights))
#         log_NC[t+1] = log_NC[t] + log_NC_increment
#         particles = particles[npr.choice(a=N, size=N, replace=True, p=weights/np.sum(weights))]
#         particles = propagate(particles, theta)
#         test_fn_est[t] = np.mean(test_fn(particles))
#     return log_NC, test_fn_est

def bootstrap_PF(data, theta, potential, propagate, test_fn, N, store_paths=False) :
    
    particles, _, log_NC, test_fn_est, _, _ = initialise(data, theta, propagate, N)
    T = np.shape(data['y'])[0]
    if store_paths :
        particles = np.zeros((N,T+1,len(data['x_0'])))
        particles[:,0] = data['x_0']
        particles[:,0] = propagate(particles[:,0], theta)
    for t in range(T) :
        if store_paths :
            weights = potential(data['y'][t], particles[:,t], theta)
        else : 
            weights = potential(data['y'][t], particles, theta)
        log_NC_increment = np.log(np.mean(weights))
        log_NC[t+1] = log_NC[t] + log_NC_increment
        resampled_idx = npr.choice(a=N, size=N, replace=True, p=weights/np.sum(weights))
        if store_paths : 
            particles[:,t] = particles[resampled_idx,t]
            particles[:,t+1] = propagate(particles[:,t], theta)
            test_fn_est[t] = np.mean(test_fn(particles[:,t+1]))
        else : 
            particles[:] = particles[resampled_idx]
            particles[:] = propagate(particles, theta)
            test_fn_est[t] = np.mean(test_fn(particles))
    return log_NC, test_fn_est, particles


################################################################################################################
######################################### ALPHA SEQUENTIAL MONTE CARLO #########################################
################################################################################################################

# alphaSMC with random connections:
def alphaSMC_random(data, theta, potential, propagate, test_fn, N, C) :
    
    particles, weights, log_NC, test_fn_est, W, resampled_idx = initialise(data, theta, propagate, N)
    T = np.shape(data['y'])[0]
    
    for t in range(T) :
        weights *= potential(particles, data['y'][t], theta)
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


def vectorized(prob_matrix, items) :
    s = prob_matrix.cumsum(axis=0)
    r = npr.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]

def alphaSMC(data, theta, potential, propagate, test_fn, N, C, alpha) :
    
    if type(alpha) == scipy.sparse.lil.lil_matrix : 
        assert N == np.shape(alpha)[0]
    else :
        A = local_exchange_graph(N, C)

    particles, weights, log_NC, test_fn_est, _, _ = initialise(data, theta, propagate, N)
    T = np.shape(data['y'])[0]
    prob_wts = np.ones((N,N))
    items = np.arange(N)
    weights = weights/N

    for t in range(T) :
        if type(alpha) == scipy.sparse.lil.lil_matrix :
            alpha_matrix = alpha
        else :
            #alpha_matrix = d_regular_graph(N, C, fix_seed=True)
            alpha_matrix = random_permute_alpha_matrix(A)
        prob_wts = scipy.sparse.diags(np.asarray(weights).reshape(-1)*potential(particles, data['y'][t], theta))*alpha_matrix
        W_bar = np.sum(prob_wts,axis=1)
        prob_matrix = (prob_wts/np.sum(prob_wts, axis=1)).T
        resampled_particles = vectorized(prob_matrix, items)
        particles[:] = particles[resampled_particles]
        log_NC[t+1] = log_NC[t] + np.log(np.sum(W_bar)) 
        weights = (W_bar/np.sum(W_bar)) 
        particles = propagate(particles, theta)
        test_fn_est[t] = np.sum(W_bar.flatten()*test_fn(particles))/np.sum(W_bar)
    
    return log_NC, test_fn_est


def d_regular_graph(N, C, fix_seed=True) :
    if fix_seed == True :
        G = nx.random_regular_graph(C, N, 12345)
    else :
        G = nx.random_regular_graph(C, N)
    A = nx.to_scipy_sparse_matrix(G)
    A = A/np.sum(A,1)
    return scipy.sparse.lil_matrix(A)

# Local exchange matrix:
def local_exchange_graph(N, C) :
    A = scipy.sparse.eye(N).tolil()/C
    for i in range(N) :
        d, d_left, d_right = 1, 1, 1
        while d < C : 
            A[i, (i+1+d_right)%N-1] = 1/C
            d_right += 1
            d += 1
            if d < C :
                A[i, (i+1-d_left)%N-1] = 1/C
                d_left += 1
                d += 1
    return A

def connectivity_const(alpha) :
    return np.sort(np.abs(np.linalg.eig(alpha)[0]))[-2]

def random_permute_alpha_matrix(alpha) :
    N = np.shape(alpha)[0]
    Id = scipy.sparse.eye(N).tolil()
    perm = npr.permutation(N)
    P = Id[perm,:]
    return (P*alpha*P.T).tolil()


################################################################################################################
############################## AUGMENTED ISLAND RESAMPLING PARTICLE FILTER #####################################
################################################################################################################

def within_island_resample(particles, theta, potential, y, A) :
    N = np.shape(particles)[0]
    items = np.arange(N)
    resampled_idxs = np.ones(N).astype(int)
    pots = potential(y, particles, theta)
    weights = A*pots
    W_out = np.sum(weights,1)
    prob_matrix = (weights.T)/W_out
    resampled_idxs = vectorized(prob_matrix, items)
    return particles[resampled_idxs], W_out

def augmented_island_resampling(particles, theta, W_out, A) :
    N = np.shape(particles)[0]
    S = np.shape(A)[0]
    idx = np.arange(N).astype(int)
    V = copy.deepcopy(W_out)
    for s in range(S) :
        V_old = copy.deepcopy(V)
        resampled_idx = copy.deepcopy(idx)
        weights = A[s]*V_old
        V = np.sum(weights,1)
        prob_matrix = (weights.T)/V
        idx = vectorized(prob_matrix, resampled_idx)
    return particles[idx]

def AIRPF(data, theta, potential, propagate, test_fn, A, store_paths=False) :
    N = np.shape(A)[1]
    T = np.shape(data['y'])[0]
    particles, _, log_NC, test_fn_est, _, _ = initialise(data, theta, propagate, N)
    if store_paths : 
        particles = np.zeros((N,T+1,len(data['x_0'])))
        particles[:,0] = data['x_0']
        particles[:,0] = propagate(particles[:,0], theta)
    for t in range(T) : 
        if store_paths :
            particles[:,t], W_out = within_island_resample(particles[:,t], theta, potential, data['y'][t], A[1])
            particles[:,t] = augmented_island_resampling(particles[:,t], theta, W_out, A)
            particles[:,t+1] = propagate(particles[:,t], theta)
            test_fn_est[t] = np.mean(test_fn(particles[:,t+1]))
        else :
            particles, W_out = within_island_resample(particles, theta, potential, data['y'][t], A[1])
            particles = augmented_island_resampling(particles, theta, W_out, A)
            particles = propagate(particles, theta)
            test_fn_est[t] = np.mean(test_fn(particles))
        log_NC[t+1] = log_NC[t] + np.log(np.mean(W_out))
    return log_NC, test_fn_est, particles

def A_(S) :
    N = 2**S
    A = np.zeros((S,N,N))
    for s in range(S) :
        A[s] = np.kron(np.kron(np.eye(2**(S-(s+1))),np.ones((2,2))/2),np.eye(2**s))
    return A







