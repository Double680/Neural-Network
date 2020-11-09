#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np

def weightInit(numInput, numOutput, initMode, distribution="normal", submode="inputNode"):
    if initMode == "zero":
        W = np.zeros((numInput, numOutput))
        
    elif initMode == "random":
        tmp = np.sqrt( 1 / numInput )
        W = np.random.normal(0, np.sqrt(tmp), W.shape)         
            
        if distribution == "uniform":
            W = np.random.uniform(-tmp, tmp, W.shape)
            
    elif initMode == "Xavier":
        tmp = np.sqrt( 2 / ( numInput + numOutput ) )
        W = np.random.normal(0, np.sqrt(tmp), (numInput, numOutput))
        
        if distribution == "uniform":
            tmp = np.sqrt( 6 / ( numInput + numOutput ) )
            W = np.random.uniform(-tmp, tmp, (numInput, numOutput))
            
    elif initMode == "He":
        n = numInput
        
        if submode == "outputNode":
            n = numOutput
            
        tmp = np.sqrt( 2 / n )
        W = np.random.normal(0, np.sqrt(tmp), (numInput, numOutput))
        
        if distribution == "uniform":
            tmp = np.sqrt( 6 / n )
            W = np.random.uniform(-tmp, tmp, (numInput, numOutput))
            
    return W

