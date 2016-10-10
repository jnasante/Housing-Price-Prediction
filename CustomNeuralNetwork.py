import numpy as np
from data_processor import data, X, y 
import math
from random import shuffle

def sigmoid(x):
  return 

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

  def train_regression(self, samples, solutions, max_loops=5000, alpha=0.001):
    hidden_layer_size = 18
    feature_count = 18
    output_layer_size = 1

    np.random.seed(10)

    weights0 = 2.0 * np.random.random( (feature_count, hidden_layer_size) ) - 1.0 
    weights1 = 2.0 * np.random.random( (hidden_layer_size, output_layer_size) ) - 1.0 

    currentError = 100.0
    
    _y = np.array([y]).T


    prevWeights0 = weights0
    prevWeights1 = weights1
    for j in range(max_loops):
      #print(weights0)
      #_X, _y = shuffle_lists(X, y)

      output1 = 1 / (1 + np.exp(-1.0 * np.dot(X, weights0)))
      output2 = 1 / (1 + np.exp(-1.0 * np.dot(output1, weights1)))

      currentError = math.sqrt(np.sum(np.square(output2 - _y)))
      if (j % 100) == 0:
        print("At iteration %d, error : %f" % (j, math.sqrt(currentError)))

      #currentError += ((y - output1) ** 2)

      delta2 = (output2 - _y) * sigmoidPrime(output2)
      delta1 = delta2.dot(weights1.T) * sigmoidPrime(output1)

      weights1 = weights1 - alpha * np.dot(output1.T, delta2)

      #print(np.sum(weights1 - prevWeights1))

      weights0 = weights0 + alpha * X.T.dot(delta1)

      #print(np.sum(weights0 - prevWeights0))

      #prevWeights0 = weights0
      #prevWeights1 = weights1


nn = NeuralNetwork()
nn.train_regression(samples=X, solutions=y)