#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np

# "mode" can be value of "logistic", "tanh", "relu" or "none" (default)
# "z" SHOULD be 2D-numpy array in size of (n, p) = (number of samples, number of features)

def activateFunc(mode="none"): # Activate Function   
    def logistic(z):
        a = 1 / ( 1 + np.exp(-z) )
        return a
    
    def tanh(z):
        a = 2 * logistic(2*z) - 1
        return a
    
    def relu(z):
        a = np.maximum(z, 0)
        return a
    
    def none(z):
        a = z
        return a
    
    if mode == "logistic":
        return logistic
    elif mode == "tanh":
        return tanh
    elif mode == "relu":
        return relu
    
    return none

def gradActivate(mode="none"): # Gradient of Activate Function
    def gradLogistic(z):
        a = activateFunc("logistic")(z)
        grad = a * (1-a)
        return grad
    
    def gradTanh(z):
        a = activateFunc("tanh")(z)
        grad = 1 - a**2
        return grad
    
    def gradRelu(z):
        grad = activateFunc("relu")(np.sign(z))
        return grad
    
    def gradNone(z):
        grad = np.ones(np.shape(z))
        
    if mode == "logistic":
        return gradLogistic
    elif mode == "tanh":
        return gradTanh
    elif mode == "relu":
        return gradRelu
    
    return gradNone

