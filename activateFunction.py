#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np

# "mode" can be value of "sigmoid", "tanh" or "relu"
# "z" SHOULD be 2D-numpy array in size of (n, p) = (number of samples, number of features)

def activateFunc(mode): # Activate Function   
    def sigmoid(z):
        a = 1 / ( 1 + np.exp(-z) )
        return a
    
    def tanh(z):
        a = 2 * sigmoid(2*z) - 1
        return a
    
    def relu(z):
        a = np.maximum(z, 0)
        return a
    
    if mode == "sigmoid":
        return sigmoid
    elif mode == "tanh":
        return tanh
    elif mode == "relu":
        return relu

def gradActivate(mode): # Gradient of Activate Function
    def gradSigmoid(z):
        a = activateFunc("sigmoid")(z)
        grad = a * (1-a)
        return grad
    
    def gradTanh(z):
        a = activateFunc("tanh")(z)
        grad = 1 - a**2
        return grad
    
    def gradRelu(z):
        grad = activateFunc("relu")(np.sign(z))
        return grad
        
    if mode == "sigmoid":
        return gradSigmoid
    elif mode == "tanh":
        return gradTanh
    elif mode == "relu":
        return gradRelu

