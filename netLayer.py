#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import activateFunction as acFunc

class netLayer():
    def __init__(self, layerMode, numNode, activeMode):
        if layerMode == "FC":
            self.numNode = numNode
            self.activeMode = activeMode
            self.activeFunc = acFunc.activateFunc(self.activeMode)
            self.gradActive = acFunc.gradActivate(self.activeMode)
    

