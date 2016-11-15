import numpy as np
import math
from random import shuffle
from scipy.special import expit
from matplotlib import pyplot as plt

def linear(x):
  return x

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
  return (2.0 / (1 + np.exp(-2.0 * x))) - 1.0

def sigmoidPrime(x):
  return x * (1.0 - x)

def tanhPrime(x):
  return 1.0 - np.square(tanh(x))

def linearPrime(x):
  return 1

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

  def __init__(self, alpha=1E-5, weights=[], w_0=0.0, decay_rate=-1, feature_count=18):
    self.alpha = alpha
    self.weights = weights
    self.w_0 = w_0
    self.decay_rate = decay_rate
    self.feature_count = feature_count

  def train_regression(self, X_train, y_train, X_val, y_val, X_test, y_test, max_loops=50, alpha=0.00001, plot_results=False):

    hidden_layer_size = self.feature_count
    output_layer_size = 1

    scalingFactor = 2.0

    weights0 = scalingFactor * np.random.random( (self.feature_count, hidden_layer_size) ) - (scalingFactor / 2.0)
    weights1 = scalingFactor * np.random.random( (hidden_layer_size, output_layer_size) ) - (scalingFactor / 2.0)

    mixedWeights0 = np.copy(weights0)
    mixedWeights1 = np.copy(weights1)

    tripleWeights0 = np.copy(weights0)
    tripleWeights1 = np.copy(weights1)

    self.trainErrors, mixedTrainErrors, tripleTrainErrors = [], [], []
    self.validationErrors, mixedValidationErrors, tripleValidationErrors = [], [], []

    vectorizedSigmoid, vectorizedTanh, vectorizedLinear = np.vectorize(sigmoid), np.vectorize(tanh), np.vectorize(linear)
    vectorizedSigmoidPrime, vectorizedTanhPrime, vectorizedLinearPrime = np.vectorize(sigmoidPrime), np.vectorize(tanhPrime), np.vectorize(linearPrime)


    for j in range(max_loops):
      currentError, currentMixedError, currentTripleError = 0.0, 0.0, 0.0
      s_x, s_y = shuffle_lists(X_train, y_train)
      for sam, sol in zip(s_x, s_y):
        
        input0 = np.dot(sam, weights0)
        mixedInput0 = np.dot(sam, mixedWeights0)
        tripleInput0 = np.dot(sam, tripleWeights0)
        
        sigmoidActivated = vectorizedSigmoid(mixedInput0[:int(hidden_layer_size / 2.0)])
        tanhActivated = vectorizedTanh(mixedInput0[int(hidden_layer_size / 2.0):])
        
        tripleSigmoidActivated = vectorizedSigmoid(tripleInput0[:int(hidden_layer_size / 3.0)])
        tripleTanhActivated = vectorizedSigmoid(tripleInput0[int(hidden_layer_size / 3.0):int(2.0 * hidden_layer_size / 3.0)])
        tripleLinearActivated = vectorizedLinear(tripleInput0[int(2.0 * hidden_layer_size / 3.0):])

        output1 = vectorizedSigmoid(input0)
        mixedOutput1 = np.append(sigmoidActivated, tanhActivated)
        tripleOutput1 = np.append(np.append(tripleSigmoidActivated, tripleTanhActivated), tripleLinearActivated)

        output2 = vectorizedSigmoid(np.dot(output1, weights1))
        mixedOutput2 = vectorizedSigmoid(np.dot(mixedOutput1, mixedWeights1))
        tripleOutput2 = vectorizedSigmoid(np.dot(tripleOutput1, tripleWeights1))

        #reg_output2 = sigmoid(np.dot(reg_sigmoidActivated, weights1))

        error = (output2 - sol) 
        mixedError = (mixedOutput2 - sol)
        tripleError = (tripleOutput2 - sol)

        currentError += (error ** 2)
        currentMixedError += (mixedError ** 2)
        currentTripleError += (tripleError ** 2)


        delta2 = error * vectorizedSigmoidPrime(output2)
        mixedDelta2 = mixedError * vectorizedSigmoidPrime(mixedOutput2)
        tripleDelta2 = tripleError * vectorizedSigmoidPrime(tripleOutput2)

        sigmoidPrimeDelta = vectorizedSigmoidPrime(sigmoidActivated)
        tanhPrimeDelta = vectorizedTanhPrime(tanhActivated)
        prime = np.append(sigmoidPrimeDelta, tanhPrimeDelta)

        tripleSigmoidPrimeDelta = vectorizedSigmoidPrime(tripleSigmoidActivated)
        tripleTanhPrimeDelta = vectorizedTanhPrime(tripleTanhActivated)
        tripleLinearPrimeDelta = vectorizedLinearPrime(tripleLinearActivated)
        triplePrime = np.append(np.append(tripleSigmoidPrimeDelta, tripleTanhPrimeDelta), tripleLinearPrimeDelta)
         
        delta1 = delta2.dot(weights1.T) * sigmoidPrime(output1)
        mixedDelta1 = mixedDelta2.dot(mixedWeights1.T) * prime
        tripleDelta1 = tripleDelta2.dot(tripleWeights1.T) * triplePrime
        
        weights1 -= np.array([alpha * (output1 * delta2)]).T
        mixedWeights1 -= np.array([alpha * (mixedOutput1 * mixedDelta2)]).T
        tripleWeights1 -= np.array([alpha * (tripleOutput1 * tripleDelta2)]).T
        
        weights0 -= alpha * np.dot(delta1, sam.T)
        mixedWeights0 -= alpha * np.dot(mixedDelta1, sam.T)
        tripleWeights0 -= alpha * np.dot(tripleDelta1, sam.T)

      if j % 10 == 0:
        self.trainErrors += [currentError]
        mixedTrainErrors += [currentMixedError]
        tripleTrainErrors += [currentTripleError]

        tripleValError, mixedValError, valError = 0.0, 0.0, 0.0
        for _x, _y in zip(X_val, y_val):

          valOutput1 = vectorizedSigmoid(np.dot(_x, weights0))
          valOutput2 = vectorizedSigmoid(np.dot(valOutput1, weights1))
          

          valMixedInput0 = np.dot(_x, mixedWeights0)
          valSigmoidActivated = vectorizedSigmoid(valMixedInput0[:int(hidden_layer_size / 2.0)])
          valTanhActivated = vectorizedTanh(valMixedInput0[int(hidden_layer_size / 2.0):])
          valMixedOutput1 = np.append(valSigmoidActivated, valTanhActivated)
          valMixedOutput2 = vectorizedSigmoid(np.dot(valMixedOutput1, mixedWeights1))

          valTripleInput0 = np.dot(sam, tripleWeights0)
          valTripleSigmoidActivated = vectorizedSigmoid(valTripleInput0[:int(hidden_layer_size / 3.0)])
          valTripleTanhActivated = vectorizedSigmoid(valTripleInput0[int(hidden_layer_size / 3.0):int(2.0 * hidden_layer_size / 3.0)])
          valTripleLinearActivated = vectorizedLinear(valTripleInput0[int(2.0 * hidden_layer_size / 3.0):])
          valTripleOutput1 = np.append(np.append(valTripleSigmoidActivated, valTripleTanhActivated), valTripleLinearActivated)
          valTripleOutput2 = vectorizedSigmoid(np.dot(valTripleOutput1, tripleWeights1))


          valError += ((_y - valOutput2) ** 2.0)
          mixedValError += ((_y - valMixedOutput2) ** 2.0)
          tripleValError += ((_y - valTripleOutput2) ** 2.0)

        print("Validation error : ", valError, mixedValError, tripleValError)
        self.validationErrors += [valError]
        mixedValidationErrors += [mixedValError]
        tripleValidationErrors += [tripleValError]

        print('Iteration: {0}\tProgress: {1}%\tError: {2}, {3}, {4}'.format(j, int(100*j/max_loops), currentError, currentMixedError, currentTripleError))

    # Write to file
    np.savetxt('tripleWeights0.txt', weights0)
    np.savetxt('tripleWeights1.txt', weights1)

    """
    testOutput1 = sigmoid(np.dot(X_test, weights0))
    testOutput2 = sigmoid(np.dot(testOutput1, weights1))
    testErr = np.sum(np.square(y_test - testOutput2))

    print("#### Test Error : ", testErr)
    """

    if (plot_results):
      plt.subplot(211)
      plt.plot(list(range(len(self.trainErrors))), self.trainErrors, 'g')
      plt.plot(list(range(len(mixedTrainErrors))), mixedTrainErrors, 'r')
      plt.plot(list(range(len(tripleTrainErrors))), tripleTrainErrors, 'b')
      plt.legend(['Regular', 'Mixed', 'Triple'])
      plt.ylabel("SSE")
      plt.xlabel("Iterations")
      plt.title("Training errors vs. iteration")

      plt.subplot(212)
      plt.plot(list(range(len(self.validationErrors))), self.validationErrors, 'g')
      plt.plot(list(range(len(mixedValidationErrors))), mixedValidationErrors, 'r')
      plt.plot(list(range(len(tripleValidationErrors))), tripleValidationErrors, 'b')
      plt.legend(['Regular', 'Mixed', 'Triple'])
      plt.ylabel("SSE")
      plt.xlabel("Iterations")
      plt.title("Validation errors vs. iteration")
    
      plt.show()   

  def predict(self, test_subject):

    hidden_layer_size = self.feature_count
    output_layer_size = 1

    tripleWeights0 = np.loadtxt('tripleWeights0.txt')
    tripleWeights1 = np.loadtxt('tripleWeights1.txt')

    vectorizedSigmoid, vectorizedTanh, vectorizedLinear = np.vectorize(sigmoid), np.vectorize(tanh), np.vectorize(linear)

    valTripleInput0 = np.dot(test_subject, tripleWeights0)
    valTripleSigmoidActivated = vectorizedSigmoid(valTripleInput0[:int(hidden_layer_size / 3.0)])
    valTripleTanhActivated = vectorizedSigmoid(valTripleInput0[int(hidden_layer_size / 3.0):int(2.0 * hidden_layer_size / 3.0)])
    valTripleLinearActivated = vectorizedLinear(valTripleInput0[int(2.0 * hidden_layer_size / 3.0):])
    valTripleOutput1 = np.append(np.append(valTripleSigmoidActivated, valTripleTanhActivated), valTripleLinearActivated)
    valTripleOutput2 = vectorizedSigmoid(np.dot(valTripleOutput1, tripleWeights1))

    return valTripleOutput2

  