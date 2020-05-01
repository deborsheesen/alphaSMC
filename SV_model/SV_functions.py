
# coding: utf-8

# In[8]:

# Calling libraries:
from __future__ import division
get_ipython().magic('matplotlib inline')
import numpy as np
import time
from scipy.stats import norm, uniform, bernoulli
from scipy.linalg import sqrtm 
from pylab import plot, show, legend
import matplotlib.pyplot as plt
import math 
import scipy
plt.rcParams['figure.figsize'] = (15.0, 3.0)
int = np.vectorize(int)


# * $X_0 = 0$, $X_n = a X_{n-1} + \mathcal{N}(0,\sigma^2)$. 
# 
# * $Y_n \sim \mathcal{N}(0,\epsilon^2 e^{X_n})$.
# 
# * $ \theta = (a, \sigma, \epsilon)$. 
# 
# * $ |a| < 1$, $\sigma > 0$, $\epsilon > 0$. 

# In[38]:

def simulate_data(theta,x_0,T) :
    X = x_0
    y = np.zeros(T)
    for t in range(T) :
        X    = theta[0]*X + theta[1]*np.random.randn(1)
        y[t] = theta[2]*np.exp(X/2)*np.random.randn(1)
    return y


# In[10]:

def potential(y,X,epsilon) :
    return norm.pdf(x=y,loc=0,scale=epsilon*np.exp(X/2))


# ### Bootstrap particle filter:
# 
# * Resample at every time step.

# In[51]:

def bootstrap_PF(n_particles,theta,x_0,y,potential,test_function) :
    
    scipy.random.seed()
    
    T = len(y)
    particles    = np.zeros(n_particles)
    particles[:] = x_0
    log_NC       = np.zeros(T+1)
    particles    = theta[0]*particles + theta[1]*np.random.randn(n_particles)
    test_function_estimate = np.zeros(T)

    for t in range(T) :
        weights     = potential(particles,y[t],theta[2]) / n_particles
        log_NC[t+1] = log_NC[t] + np.log(np.sum(weights))
        particles   = particles[np.random.choice(a=n_particles,size=n_particles,replace=True,p=weights/np.sum(weights))]
        particles   = theta[0]*particles + theta[1]*np.random.randn(n_particles)
        
        test_function_estimate[t] = np.mean(test_function(particles))
        
    return log_NC[1::], test_function_estimate


# ### $\alpha$SMC with random connections:

# In[61]:

def alphaSMC_random(n_particles,theta,x_0,y,n_connections,potential,test_function) :
    
    scipy.random.seed()
    
    T = len(y)
    particles       = np.zeros((n_particles,1))
    particles[:,0]  = x_0
    weights         = np.ones(n_particles)
    log_NC          = np.zeros(T)
    particles[:,0]  = theta[0]*particles[:,0] + theta[1]*np.random.randn(n_particles)
    test_function_estimate = np.zeros(T)
    
    W = np.ones(n_particles)
    w = np.ones(n_particles)
    normalized_weights = np.ones(n_particles) 
    resampled_index = [1]*n_particles

    for t in range(T) :
        w[:] = weights * potential( particles[:,0], y[t], theta[2] )
        for particle in range(n_particles) :
            connections = np.random.choice(a=n_particles,size=n_connections,replace=False)
            W[particle] = np.mean(w[connections])
            resampled_index[particle] = np.random.choice(a=connections,p=w[connections]/np.sum(w[connections]))
        particles[:,0] = particles[resampled_index,0]
        weights[:]     = W
        log_NC[t]      = np.log(np.mean(weights))
        particles[:,0] = theta[0]*particles[:,0] + theta[1]*np.random.randn(n_particles)
        normalized_weights[:] = weights/np.sum(weights)
        
        test_function_estimate[t] = np.sum(normalized_weights*test_function(particles[:,0]))
    
    return log_NC, test_function_estimate


# In[ ]:



