import numpy as np
from data_processor import data, X, y 
import math
from random import shuffle
from scipy.special import expit
from matplotlib import pyplot as plt

def sigmoid(x):
  output = 1.0 / (1.0 + np.exp(-x))
  return output

def sigmoidPrime(x):
  retval = x * (1.0 - x)
  return retval

def featureScale(x):
  means = np.mean(x, axis=0)
  devs = np.std(x, axis=0)
  return np.array((x - means) / devs)

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

  def train_regression(self, samples, solutions, max_loops=1000, alpha=0.00001):
    
    hidden_layer_size = 18
    feature_count = 18
    output_layer_size = 1

    scalingFactor = 2.0

    weights0 = scalingFactor * np.random.random( (feature_count, hidden_layer_size) ) - (scalingFactor / 2.0)
    weights1 = scalingFactor * np.random.random( (hidden_layer_size, output_layer_size) ) - (scalingFactor / 2.0)

    
    errors = []
    for j in range(max_loops):
      currentError = 0.0
      for sam, sol in zip(samples, solutions):
        output1 = sigmoid(np.dot(sam, weights0))
        output2 = sigmoid(np.dot(output1, weights1))

        error = (output2 - sol) 
        currentError += error ** 2

        delta2 = error * sigmoidPrime(output2)
        delta1 = delta2.dot(weights1.T) * sigmoidPrime(output1)
        
        weights1 -= np.array([alpha * (output1 * delta2)]).T
        weights0 -= alpha * sam.T.dot(delta1)

      if j % 10 == 0:
        errors += [currentError]
        print("At iteration %d, error : %f" % (j, math.sqrt(currentError)))

    plt.plot(list(range(0, max_loops, 10)), errors)
    plt.ylabel("SSE")
    plt.xlabel("Iterations")
    plt.title("Training errors vs. iteration")
    plt.show()
        
      

      

nn = NeuralNetwork()
scaledX, scaledY = featureScale(X), featureScale(y)


nn.train_regression(samples=scaledX, solutions=scaledY.T)
