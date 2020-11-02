import numpy as np

# "mode" can be value of "reg", "biClassify", "mulClassify"
# "a", "y" SHOULD be 2D-numpy array in size of (n, p) = (number of samples, number of features)
# "loss" is 1D-numpy array in size of n = number of samples
# "z" SHOULD be 2D-numpy array in size of (n, p) = (number of samples, number of features)

def lossFunc(mode):
    def lossReg(a, y):
        loss = np.sum( (a - y)**2, axis=1) / 2
        return loss
    
    def lossBiClassify(a, y):
        loss = np.sum( -(y*np.log(a) + (1-y)*np.log(1-a) ), axis=1)
        return loss
    
    def lossMulClassify(a, y):
        loss = np.sum( -y*np.log(a), axis=1)
        return loss
    
    if mode == "reg":
        return lossReg
    elif mode == "biClassify":
        return lossBiClassify
    elif mode == "mulClassify":
        return lossMulClassify
    
def predictFunc(mode): 
    def noChange(z):
        a = z
        return a
    
    def logistic(z):
        a = 1 / ( 1 + np.exp(-z) )
        return a
    
    def softmax(z):
        tmp = np.sum( np.exp(z), axis=1)
        a = np.exp(z) / tmp.reshape(tmp.shape[0], 1)
        return a
    
    if mode == "reg":
        return noChange
    elif mode == "biClassify":
        return logistic
    elif mode == "mulClassify":
        return softmax    

def gradZ(a, y):
    return a - y
