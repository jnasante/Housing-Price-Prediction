import numpy as np
from data_processor import data, X, y, cross_validation
import math
from random import shuffle
from scipy.special import expit
from matplotlib import pyplot as plt

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
  return (2.0 / (1 + np.exp(-2.0 * x))) - 1.0

def sigmoidPrime(x):
  return x * (1.0 - x)

def tanhPrime(x):
  return 1.0 - np.square(tanh(x))

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

  def train_regression(self, X_train, y_train, X_val, y_val, X_test, y_test, max_loops=500, alpha=0.0001):
    
    hidden_layer_size = 18
    feature_count = 18
    output_layer_size = 1

    scalingFactor = 2.0

    weights0 = scalingFactor * np.random.random( (feature_count, hidden_layer_size) ) - (scalingFactor / 2.0)
    weights1 = scalingFactor * np.random.random( (hidden_layer_size, output_layer_size) ) - (scalingFactor / 2.0)

    mixedWeights0 = np.copy(weights0)
    mixedWeights1 = np.copy(weights1)

    tripleWeights0 = np.copy(weights0)
    tripleWeights1 = np.copy(weights1)

    trainErrors, mixedTrainErrors, tripleTrainErrors = [], [], []
    validationErrors, mixedValidationErrors, tripleValidationErrors = [], [], []

    vectorizedSigmoid, vectorizedTanh = np.vectorize(sigmoid), np.vectorize(tanh)
    vectorizedSigmoidPrime, vectorizedTanhPrime = np.vectorize(sigmoidPrime), np.vectorize(tanhPrime)

    for j in range(max_loops):
      currentError, currentMixedError = 0.0, 0.0
      s_x, s_y = shuffle_lists(X_train, y_train)
      for sam, sol in zip(s_x, s_y):

        input0 = np.dot(sam, weights0)
        mixedInput0 = np.dot(sam, mixedWeights0)
        
        sigmoidActivated = vectorizedSigmoid(mixedInput0[:int(hidden_layer_size / 2.0)])
        tanhActivated = vectorizedTanh(mixedInput0[int(hidden_layer_size / 2.0):])

        output1 = vectorizedSigmoid(input0)
        mixedOutput1 = np.append(sigmoidActivated, tanhActivated)

        output2 = vectorizedSigmoid(np.dot(output1, weights1))
        mixedOutput2 = vectorizedSigmoid(np.dot(mixedOutput1, mixedWeights1))

        #reg_output2 = sigmoid(np.dot(reg_sigmoidActivated, weights1))

        error = (output2 - sol) 
        mixedError = (mixedOutput2 - sol)

        currentError += (error ** 2)
        currentMixedError += (mixedError ** 2)


        delta2 = error * vectorizedSigmoidPrime(output2)
        mixedDelta2 = mixedError * vectorizedSigmoidPrime(mixedOutput2)

        sigmoidPrimeDelta = vectorizedSigmoidPrime(sigmoidActivated)
        tanhPrimeDelta = vectorizedTanhPrime(tanhActivated)
        prime = np.append(sigmoidPrimeDelta, tanhPrimeDelta)
         
        delta1 = delta2.dot(weights1.T) * sigmoidPrime(output1)
        mixedDelta1 = mixedDelta2.dot(mixedWeights1.T) * prime
        
        weights1 -= np.array([alpha * (output1 * delta2)]).T
        mixedWeights1 -= np.array([alpha * (mixedOutput1 * mixedDelta2)]).T
        
        weights0 -= alpha * np.dot(delta1, sam.T)
        mixedWeights0 -= alpha * np.dot(mixedDelta1, sam.T)

      if j % 10 == 0:
        trainErrors += [currentError]
        mixedTrainErrors += [currentMixedError]

        

        mixedValError, valError = 0.0, 0.0
        for _x, _y in zip(X_val, y_val):

          valOutput1 = vectorizedSigmoid(np.dot(_x, weights0))
          valOutput2 = vectorizedSigmoid(np.dot(valOutput1, weights1))
          valError += ((_y - valOutput2) ** 2.0)

          valMixedInput0 = np.dot(_x, mixedWeights0)
          valSigmoidActivated = vectorizedSigmoid(valMixedInput0[:int(hidden_layer_size / 2.0)])
          valTanhActivated = vectorizedTanh(valMixedInput0[int(hidden_layer_size / 2.0):])
          valMixedOutput1 = np.append(valSigmoidActivated, valTanhActivated)
          valMixedOutput2 = vectorizedSigmoid(np.dot(valMixedOutput1, mixedWeights1))

          mixedValError += ((_y - valMixedOutput2) ** 2.0)

        print("Validation error : ", valError, mixedValError)

        validationErrors += [valError]
        mixedValidationErrors += [mixedValError]


        print("At iteration %d, error : %f" % (j, currentError))

    plt.subplot(211)
    plt.plot(list(range(0, max_loops, 10)), trainErrors, 'g')
    plt.plot(list(range(0, max_loops, 10)), mixedTrainErrors, 'r')
    plt.legend(['Regular Train', 'Mixed Train'])
    plt.ylabel("SSE")
    plt.xlabel("Iterations")
    plt.title("Training errors vs. iteration (alpha = {0})".format(alpha))

    plt.subplot(212)
    plt.plot(list(range(0, max_loops, 10)), validationErrors, 'g')
    plt.plot(list(range(0, max_loops, 10)), mixedValidationErrors, 'r')
    plt.legend(['Regular', 'Mixed'])
    plt.ylabel("SSE")
    plt.xlabel("Iterations")
    plt.title("Validation errors vs. iteration (alpha : {0})".format(alpha))


    testErr, mixedTestError = 0.0, 0.0
    for _x, _y in zip(X_test, y_test):

          testOutput1 = vectorizedSigmoid(np.dot(_x, weights0))
          testOutput2 = vectorizedSigmoid(np.dot(testOutput1, weights1))
          testErr += ((_y - testOutput2) ** 2.0)

          mixedTestInput0 = np.dot(_x, mixedWeights0)
          mixedTestSigmoidActivated = vectorizedSigmoid(mixedTestInput0[:int(hidden_layer_size / 2.0)])
          mixedTestTanhActivated = vectorizedTanh(mixedTestInput0[int(hidden_layer_size / 2.0):])
          mixedTestOutput1 = np.append(mixedTestSigmoidActivated, mixedTestTanhActivated)
          mixedTestOutput2 = vectorizedSigmoid(np.dot(mixedTestOutput1, mixedWeights1))
          mixedTestError += ((_y - mixedTestOutput2) ** 2.0)    

    print("#### Test Error : ", testErr, mixedTestError)
    
    plt.show()
        
      

      

nn = NeuralNetwork()
scaledX, scaledY = featureScale(X), featureScale(y)
X_train, y_train, X_val, y_val, X_test, y_test = cross_validation(scaledX, scaledY)

nn.train_regression(X_train, y_train.T, X_val, y_val.T, X_test, y_test.T)
