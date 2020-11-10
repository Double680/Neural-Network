#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

# "initMode" can be the value of "zero", "random" (default), "Xavier" or "He"
# when "initMode" is not "zero", "distribution" can be the value of "normal" (default) or "uniform"
# when "initMode" is "He", "submode" can be "inputNode" (default) or "outputNode"

def weightInit(numInput, numOutput, initMode="random", distribution="normal", submode="inputNode"):
    
    # "random" initialization
    tmp = np.sqrt( 1 / numInput )
    weight = np.random.normal(0, np.sqrt(tmp), (numInput, numOutput))                 
    if distribution == "uniform":
        weight = np.random.uniform(-tmp, tmp, (numInput, numOutput))
    
    if initMode == "zero":
        weight = np.zeros((numInput, numOutput))
   
    elif initMode == "Xavier":
        tmp = np.sqrt( 2 / ( numInput + numOutput ) )
        weight = np.random.normal(0, np.sqrt(tmp), (numInput, numOutput))
        
        if distribution == "uniform":
            tmp = np.sqrt( 6 / ( numInput + numOutput ) )
            weight = np.random.uniform(-tmp, tmp, (numInput, numOutput))
            
    elif initMode == "He":
        n = numInput
        if submode == "outputNode":
            n = numOutput
            
        tmp = np.sqrt( 2 / n )
        weight = np.random.normal(0, np.sqrt(tmp), (numInput, numOutput))
        
        if distribution == "uniform":
            tmp = np.sqrt( 6 / n )
            weight = np.random.uniform(-tmp, tmp, (numInput, numOutput))
    
    return weight

