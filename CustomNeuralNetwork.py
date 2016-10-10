import numpy as np
from data_processor import data, X, y 
import math
from random import shuffle

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoidPrime(x):
  return x * (1 - x)

def shuffle_lists(a, b):
  _a = []
  _b = []
  idx = list(range(len(a)))
  shuffle(idx)
  for i in idx:
    _a.append(a[i])
    _b.append(b[i])
    
  return np.array(_a), np.array(_b)

class NeuralNetwork():

  def __init__(self, alpha=1E-5, weights=[], w_0=0.0, decay_rate=-1):
    self.alpha = alpha
    self.weights = weights
    self.w_0 = w_0
    self.decay_rate = decay_rate
    
  def predict_regression(self, samples):
    return np.dot(samples, self.weights) + self.w_0

  def train_regression(self, samples, solutions, max_loops=50000, alpha=0.01):
    hidden_layer_size = 18
    feature_count = 18
    output_layer_size = 1

    np.random.seed(10)

    weights0 = 2.0 * np.random.random( (feature_count,hidden_layer_size) ) - 1.0 
    weights1 = 2.0 * np.random.random( (hidden_layer_size, output_layer_size) ) - 1.0 

    currentError = 100.0
    numSamples = len(y)

    for j in range(max_loops):
      if (j % 1000) == 0:
        print("At iteration %d, error : %f" % (j, math.sqrt(currentError)))

      currentError = 0.0
      _X, _y = shuffle_lists(X, y)
      for x, label in zip(_X, _y):

        output0 = sigmoid(weights0.dot(x))
        output1 = sigmoid(output0.dot(weights1))

        currentError += ((label - output1) ** 2)

        delta1 = (label - output1) * sigmoidPrime(output1)
        delta0 = delta1.dot(weights1.T) * sigmoidPrime(output0)

        weights1 += (alpha * output0.T.dot(delta1))
        weights0 += (alpha * x.T.dot(delta0))






nn = NeuralNetwork()
nn.train_regression(samples=X, solutions=y)