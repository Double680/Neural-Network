# Neural-Network

Training Programs of Neural Network.
Started on Nov 2nd, 2020

## Loss Function

We suppose that the final loss $J=\sum_{i=1}^n loss_i$.

### Regression 

$$loss=\frac{1}{2}(a-y)^2$$
$$a=z$$

### Binary Classification

$$loss=-(y\ln a+(1-y)\ln(1-a))$$
$$a=Logistic(z)=\frac{1}{1+e^{-z}}$$

### Multiple Classification

$$loss=-\sum_{j=1}^p y_j\ln a_j$$
$$a=Softmax(z),a_k=\frac{e^k}{\sum_{j=1}^pe^j},\forall k$$

## Activate Function

We can use `sigmoid`, `tanh` and `relu` now.
 
