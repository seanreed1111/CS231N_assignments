import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  loss_elem=np.zeros(X.shape[0])
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    class_score=np.zeros(W.shape[1])
    temp_grad=np.zeros(W.shape)
    for j in range(W.shape[1]):
      class_score[j]= X[i,:].dot(W[:,j])
      temp_grad[:,j]+=X[i]*np.exp(class_score[j])

    dW[:,y[i]]+=X[i]-temp_grad[:,y[i]]/np.sum(np.exp(class_score))
    dW[:,0:y[i]]-=temp_grad[:,0:y[i]]/np.sum(np.exp(class_score))
    dW[:, y[i]+1:]-=temp_grad[:,y[i]+1:]/np.sum(np.exp(class_score))
    max_class_score=np.max(class_score)
    class_score -=max_class_score
    loss_elem[i]=-np.log(np.exp(class_score[y[i]])/np.sum((np.exp(class_score)))) # loss over one data sample
  loss =np.sum(loss_elem)/X.shape[0]+reg*np.sum(np.square(W)) # average over all the examples
  dW/=X.shape[0]
  dW=-dW
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  prob=np.zeros(X.shape[0])
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  class_scores=X.dot(W)
  bin=np.zeros(class_scores.shape)
  prob=np.exp(class_scores)
  loss=-np.sum(np.log(prob[np.arange(X.shape[0]),y]/np.sum(prob,axis=1)))
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss/=X.shape[0]
  loss+=reg*np.sum(np.square(W))
  prob_norm = np.divide(prob, np.sum(prob, axis=1).reshape(X.shape[0],-1))

  dW = -prob_norm.T.dot(X)  # here we have computed the transpose of the dW matrix ( check the dimensions)
  dW= dW.T
  bin[np.arange(X.shape[0]),y]=1
  dW= - (dW+ (X.T.dot(bin)))/X.shape[0]
  dW+= reg*W
  return loss, dW

