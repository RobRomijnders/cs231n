import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # Initialize filter and biases
    C,H,W = input_dim
    F = num_filters
    
    #Conv layer
    # Input size = (N,C,H,W)
    # Output size = (N,F,Hc,Wc)
    F_H = filter_size
    F_W = filter_size
    S = 1  #Stride
    P = (filter_size - 1)/2  # padding
    
    Hc = 1 + (H+2*P - F_H)/S
    Wc = 1 + (W+2*P - F_W)/S
   
    self.params['W1'] = weight_scale*np.random.randn(F,C,F_H,F_W)
    self.params['b1'] = np.zeros(F)
    
    #Pool layer
    # Input : (N,F,Hc,Wc)
    # Output : (N,F,Hp,Wp)
    width_pool = 2
    height_pool = 2
    stride_pool = 2
    Hp = (Hc - height_pool) / stride_pool + 1
    Wp = (Wc - width_pool) / stride_pool + 1
    
    # Hidden Affine layer
    # Input (N,F,Hp,Wp)(N,F*Hp*Wp)
    # Output (N,Hh)
    Hh = hidden_dim
    
   
    self.params['W2'] = weight_scale*np.random.randn(F*Hp*Wp,Hh)
    self.params['b2'] = np.zeros(Hh)
    
    #Output affine layer
    # Input (N,Hh)
    # Output (N,num_classes)
    
    
    
    self.params['W3'] = weight_scale*np.random.randn(Hh,num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out, cache_aff1 = affine_relu_forward(out, W2, b2)
    out, cache_aff2 = affine_forward(out, W3, b3)
    scores = out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    dout,dW3,db3 = affine_backward(dout, cache_aff2)
    dout,dW2,db2 = affine_relu_backward(dout, cache_aff1)
    dout,dW1,db1 = conv_relu_pool_backward(dout, cache_conv)
    
    reg = self.reg
    loss += 0.5*reg*(np.sum(W3**2)+np.sum(W2**2)+np.sum(W1**2))
    
    grads['W3'] = dW3 + reg*W3
    grads['b3'] = db3
    grads['W2'] = dW2 + reg*W2
    grads['b2'] = db2
    grads['W1'] = dW1 + reg*W1
    grads['b1'] = db1
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
