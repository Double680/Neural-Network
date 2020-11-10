#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# "mode" can be value of "biC", "mulC" or "reg" (default)
# "a", "y" SHOULD be 2D-numpy array in size of (n, p) = (number of samples, number of features)
# "loss" is 1D-numpy array in size of n = number of samples
# "z" SHOULD be 2D-numpy array in size of (n, p) = (number of samples, number of features)

def lossFunc(mode="reg"):
    def lossBiC(a, y):
        loss = np.sum( -(y*np.log(a) + (1-y)*np.log(1-a) ), axis=1)
        return loss
    
    def lossMulC(a, y):
        loss = np.sum( -y*np.log(a), axis=1)
        return loss
    
    def lossReg(a, y):
        loss = np.sum( (a - y)**2, axis=1) / 2
        return loss
    
    if mode == "biC":
        return lossBiC
    elif mode == "mulC":
        return lossMulC
    
    return lossReg
    
def predictFunc(mode="reg"): 
    def logistic(z):
        a = 1 / ( 1 + np.exp(-z) )
        return a
    
    def softmax(z):
        tmp = np.sum( np.exp(z), axis=1)
        a = np.exp(z) / tmp.reshape(tmp.shape[0], 1)
        return a
    
    def identity(z):
        a = z
        return a
    
    if mode == "biC":
        return logistic
    elif mode == "mulC":
        return softmax    
    
    return identity
    

