# Calling libraries:
from __future__ import division
import numpy as np, numpy.random as npr
from tqdm import trange
from PF import *

def propose_RW(current, scale) :
    return current + scale*npr.randn(len(current))

def update(current, proposed, log_ratio) :
    if np.log(npr.rand()) < log_ratio :
        return current, 1 
    else :
        return proposed, 0
    
def acceptance_ratio_RW(current, proposed, data, log_prior) :
    current_theta, current_ll = current['theta'], current['ll']
    proposed_theta, proposed_ll = proposed['theta'], proposed['ll']
    return (log_prior(proposed_theta) + proposed_ll)-(log_prior(current_theta) + current_ll)

    
def pm_MH(theta_init, data, n_iter, propose, acceptance_ratio, scale, potential, propagate, N, log_prior) :
    
    def test_fn(x) :
        return x
    accepted = 0
    
    x_0, y = data['x_0'], data['y']
    theta_dim = len(theta_init)
    chain = np.zeros((n_iter+1, theta_dim))
    chain[0] = theta_init
    current_ll = bootstrap_PF(data, chain[0], potential, propagate, test_fn, N)[0][-1]
    
    for i in trange(n_iter) :
        current_theta = chain[i]
        current = dict(theta=current_theta, ll=current_ll)
        proposed_theta = propose(current_theta, scale)
        proposed_ll = bootstrap_PF(data, proposed_theta, potential, propagate, test_fn, N)[0][-1]
        proposed = dict(theta=proposed_theta, ll=proposed_ll)
        log_ratio = acceptance_ratio_RW(current, proposed, data, log_prior)
        
        current, acc = update(current, proposed, log_ratio)
        accepted += acc
        chain[i+1] = current['theta']
    
    return chain, accepted/n_iter


