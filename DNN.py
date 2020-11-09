#!/usr/bin/env python
# coding: utf-8

# In[4386]:


import numpy as np
import lossFunction as lossF
from netLayer import netLayer
from weightInit import weightInit

class deepNeuralNetwork():
    def __init__(self, x, y, taskMode):
        self.x = x                                             # 2D numpy array
        self.y = y
        self.taskMode = taskMode                               # "reg", "biC" or "mulC"
        self.numSample = self.x.shape[0]
        
        # WeightMatrix Initialization
        self.initMode = "zero"
        self.initDistribution = "normal"
        self.initSubmode = "inputNode"
        
        self.numLayer = 0
        self.layerSet = []
        self.weightSet = []
        self.biasSet = []
        
        self.learningRate = 0.01
        
        self.predictFunc = lossF.predictFunc(self.taskMode)
        self.lossFunc = lossF.lossFunc(self.taskMode)
        self.finalLoss = "none"
        self.a = "none"
            
    def addLayer(self, layerMode, numNode, activeMode):
        lastLayerNode = self.x.shape[1]
        if self.numLayer != 0:
            lastLayerNode = self.layer[self.numLayer - 1].numNode
        
        self.numLayer += 1
        self.layerSet.append(netLayer(layerMode, numNode, activeMode)) 
            
        W = weightInit(lastLayerNode, numNode, self.initMode,
                       self.initDistribution, self.initSubmode)
        B = np.random.normal(0, 1, (1, numNode))
        self.weightSet.append(W)
        self.biasSet.append(B)  
        
    def removeLayer(self):
        if self.numLayer != 0:
            self.numLayer -= 1
            self.layerSet.pop()
            self.weightSet.pop()
            self.biasSet.pop()
        else:
            print("No more layer remained!")
        
    def setInitMode(self, initMode, distribution="normal", submode="inputNode"):
        self.initMode = initMode
        self.distribution = distribution
        self.submode = submode
        
        lastLayerNode = self.x.shape[1]
        
        for i in range(self.numLayer):
            self.weightSet[i] = weightInit(lastLayerNode, self.layerSet[i].numNode, 
                                           self.initMode, self.initDistribution, self.initSubmode)
            lastLayerNode = self.layerSet[i].numNode
        
    def setLearningRate(self, learningRate):
        if (isinstance(learningRate, int) or isinstance(learningRate, float)) and learningRate > 0:
            self.learningRate = learningRate
        else:
            print("Illegal learning rate!")
        
    def forward(self):
        self.layerOutputSet = []
        self.layerActivatedOutputSet = []
        
        layerInput = self.x
        
        for i in range(self.numLayer):
            layerOutput = np.dot(layerInput, self.weightSet[i]) + self.biasSet[i]
            self.layerOutputSet.append(layerOutput)
            activatedOutput = self.layerSet[i].activeFunc(layerOutput)
            self.layerActivatedOutputSet.append(activatedOutput)
            layerInput = activatedOutput
        
        self.a = self.predictFunc(layerInput)
        self.loss = self.lossFunc(self.a, self.y)
        self.finalLoss = np.sum(self.loss)
        
    def backPropagation(self):
        self.layerGradOutput = []
        self.layerGradW = []
        self.layerGradB = []
        
        gradZ = self.a - self.y
        
        for i in range(self.numLayer-1, -1, -1):
            X = self.layerActivatedOutputSet[i-1]
            if i == 0:
                X = self.x
            self.layerGradOutput.insert(0, gradZ)
            self.layerGradW.insert(0, np.dot(X.T, gradZ))
            self.layerGradB.insert(0, np.dot(np.ones((1, self.numSample)), gradZ))
            if i > 0:
                gradA = np.dot(gradZ, W.T)
                gradZ = gradA * self.layerSet[i-1].gradActive(self.layerOutputSet[i-1])
        
        for i in range(self.numLayer):
            self.weightSet[i] -= self.learningRate * self.layerGradW[i]
            self.biasSet[i] -= self.learningRate * self.layerGradB[i]
            
    def train(self, eps):
        cnt = 0
        while cnt < 10000:
            self.forward()
            if self.finalLoss < eps:
                break
            self.backPropagation()
            cnt += 1
            
    def getResult(self):
        print(self.weightSet)
        print(self.biasSet)
        print(self.finalLoss)
        


# In[4492]:


X = np.random.uniform(-10, 10, (100, 2))
W = np.array([[2],[-1]])
B = np.array([[1]])
Y = np.dot(X, W) + B + np.random.normal(0, 1, (100, 1))
net = deepNeuralNetwork(X, Y, "reg")
net.addLayer("FC", 1, "none")
net.setInitMode("Xavier")
net.setLearningRate(5e-5)


# In[4493]:


net.train(10)
net.getResult()

