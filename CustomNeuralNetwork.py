import numpy as np
from data_processor import data, X, y, cross_validation
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

  def train_regression(self, X_train, y_train, X_val, y_val, X_test, y_test, max_loops=500, alpha=0.00001):
    
    hidden_layer_size = 18
    feature_count = 18
    output_layer_size = 1

    scalingFactor = 2.0

    weights0 = scalingFactor * np.random.random( (feature_count, hidden_layer_size) ) - (scalingFactor / 2.0)
    weights1 = scalingFactor * np.random.random( (hidden_layer_size, output_layer_size) ) - (scalingFactor / 2.0)

    
    trainErrors = []
    validationErrors = []
    for j in range(max_loops):
      currentError = 0.0
      s_x, s_y = shuffle_lists(X_train, y_train)
      for sam, sol in zip(s_x, s_y):
        output1 = sigmoid(np.dot(sam, weights0))
        output2 = sigmoid(np.dot(output1, weights1))

        error = (output2 - sol) 
        currentError += (error ** 2)

        delta2 = error * sigmoidPrime(output2)
        delta1 = delta2.dot(weights1.T) * sigmoidPrime(output1)
        
        weights1 -= np.array([alpha * (output1 * delta2)]).T
        weights0 -= alpha * sam.T.dot(delta1)

      if j % 10 == 0:
        trainErrors += [currentError]

        valOutput1 = sigmoid(np.dot(X_val, weights0))
        valOutput2 = sigmoid(np.dot(valOutput1, weights1))
        valErr = np.sum(np.square(y_val - valOutput2))
        print("Validation error : ", valErr)
        validationErrors += [valErr]


        print("At iteration %d, error : %f" % (j, currentError))

    plt.subplot(211)
    plt.plot(list(range(0, max_loops, 10)), trainErrors, 'g')
    plt.legend(['Train'])
    plt.ylabel("SSE")
    plt.xlabel("Iterations")
    plt.title("Training errors vs. iteration")

    plt.subplot(212)
    plt.plot(list(range(0, max_loops, 10)), validationErrors, 'b')
    plt.legend(['Validate'])
    plt.ylabel("SSE")
    plt.xlabel("Iterations")
    plt.title("Validation errors vs. iteration")

    testOutput1 = sigmoid(np.dot(X_test, weights0))
    testOutput2 = sigmoid(np.dot(testOutput1, weights1))
    testErr = np.sum(np.square(y_test - testOutput2))

    print("#### Test Error : ", testErr)
    
    plt.show()
        
      

      

nn = NeuralNetwork()
scaledX, scaledY = featureScale(X), featureScale(y)
X_train, y_train, X_val, y_val, X_test, y_test = cross_validation(scaledX, scaledY)

nn.train_regression(X_train, y_train.T, X_val, y_val.T, X_test, y_test.T)
