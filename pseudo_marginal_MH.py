# Calling libraries:
from __future__ import division
import numpy as np, numpy.random as npr
from tqdm import trange
from PF import *

def propose_RW(current, scale, update_dims=[]) :
    proposed = current 
    proposed[update_dims] = (current + scale*npr.randn(len(current)))[update_dims]
    return proposed

def update(current, proposed, log_ratio) :
    if np.log(npr.rand()) < log_ratio :
        return current, 1 
    else :
        return proposed, 0
    
def acceptance_ratio_RW(current, proposed, data, log_prior) :
    current_theta, current_ll = current['theta'], current['ll']
    proposed_theta, proposed_ll = proposed['theta'], proposed['ll']
    return (log_prior(proposed_theta) + proposed_ll)-(log_prior(current_theta) + current_ll)

def adapt_scale(i, n_iter, chain, mu, mu2, scale, adapt=False, eps=1e-3) :
    theta_dim = len(mu)
    var = scale**2
    if adapt : 
        if i >= int(0.2*n_iter) :
            if i == int(0.2*n_iter) :
                mu = np.mean(chain[:i+1],0)
                mu2 = np.mean(chain[:i+1]**2,0)
            elif i > int(0.2*n_iter) :
                mu = (i*mu + chain[i+1])/(i+1)
                mu2 = (i*mu2 + chain[i+1]**2)/(i+1)
            var = mu2 - mu**2
            scale = np.sqrt(2.4**2/theta_dim*(var + eps)/3)
    return scale, mu, mu2

    
def pm_MH(theta_init, data, n_iter, propose, acceptance_ratio, scale, log_potential, propagate, N, log_prior, adapt=False, update_dims=[]) :
    
    def test_fn(x) :
        return x
    accepted = 0
    
    x_0, y = data['x_0'], data['y']
    theta_dim = len(theta_init)
    chain = np.zeros((n_iter+1, theta_dim))
    chain[0] = theta_init
    current_ll = bootstrap_PF(data, chain[0], log_potential, propagate, test_fn, N)[0][-1]
    mu, mu2 = np.zeros(theta_dim), np.zeros(theta_dim)
    scales = np.zeros((n_iter, theta_dim))
    
    for i in trange(n_iter) :
        current_theta = chain[i]
        current = dict(theta=current_theta, ll=current_ll)
        proposed_theta = propose(current_theta, scale, update_dims)
        proposed_ll = bootstrap_PF(data, proposed_theta, log_potential, propagate, test_fn, N)[0][-1]
        proposed = dict(theta=proposed_theta, ll=proposed_ll)
        log_ratio = acceptance_ratio_RW(current, proposed, data, log_prior)
        
        current, acc = update(current, proposed, log_ratio)
        accepted += acc
        chain[i+1] = current['theta']
        scale, mu, mu2 = adapt_scale(i, n_iter, chain, mu, mu2, scale, adapt)
        scales[i] = scale
    
    return chain, accepted/n_iter, scales


