#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from .activateFunction import *

class netLayer():
    def __init__(self, layerMode, numNode, activeMode):
        if layerMode == "FC":
            if isinstance(numNode, int) and numNode > 0:
                self.numNode = numNode
                self.activeMode = activeMode
                self.activeFunc = activateFunc(self.activeMode)
                self.gradActive = gradActivate(self.activeMode)
            else:
                print("Illegal number of nodes!")
                return
        else:
            print("Illegal layer mode!")
            

