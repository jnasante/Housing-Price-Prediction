import numpy as np
import math
from random import shuffle
from scipy.special import expit
from matplotlib import pyplot as plt


class Activation():
  @staticmethod
  def linear(x):
    return x

  @staticmethod
  def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

  @staticmethod
  def tanh(x):
    return (2.0 / (1 + np.exp(-2.0 * x))) - 1.0

  @staticmethod
  def sigmoidPrime(x):
    return x * (1.0 - x)

  @staticmethod
  def tanhPrime(x):
    return 1.0 - np.square(Activation.tanh(x))

  @staticmethod
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
    # Neural Net constants
    self.alpha = alpha
    self.weights = weights
    self.w_0 = w_0
    self.decay_rate = decay_rate

    # Other constants
    self.feature_count = feature_count
    self.hidden_layer_size = feature_count
    self.output_layer_size = 1
    self.halfway_point = int(self.hidden_layer_size / 2.0)
    self.thirdway_point = int(self.hidden_layer_size / 3.0)

  def train_regression(self, X_train, y_train, X_val, y_val, X_test, y_test, max_loops=50, alpha=0.00001, plot_results=False, num_hidden_layers=1):


    weights = [self.initialize_weight_layer(self.feature_count, self.hidden_layer_size), self.initialize_weight_layer(self.hidden_layer_size, self.output_layer_size)]
    mixedWeights = np.copy(weights)
    tripleWeights = np.copy(weights)

    self.trainErrors, mixedTrainErrors, tripleTrainErrors = [], [], []
    self.validationErrors, mixedValidationErrors, tripleValidationErrors = [], [], []

    self.initialize_vectors()

    for j in range(max_loops):
      currentError, currentMixedError, currentTripleError = 0.0, 0.0, 0.0
      s_x, s_y = shuffle_lists(X_train, y_train)

      for sam, sol in zip(s_x, s_y):
        
        # Hidden layer after activation
        reg_input_raw, mixed_input_raw, triple_input_raw = sam, sam, sam
        mixedActivated, tripleActivated = [], []

        # For each hidden layer hidden (apply mixing to all layers except output)
        for l in range(num_hidden_layers):
          # Dot with weights first
          reg_output_dotted.append(np.dot(reg_input_raw, weights[l]))
          mixed_output_dotted.append(np.dot(mixed_input_raw, mixedWeights[l]))
          triple_output_dotted.append(np.dot(triple_input_raw, tripleWeights[l]))

          # Activate them
          regActivated.append(self.activate_reg(reg_output_dotted[l]))
          mixedActivated.append(self.activate_mixed(mixed_output_dotted[l]))
          tripleActivated.append(self.activate_triple(triple_output_dotted[l]))

          reg_input_raw = regActivated[l]
          mixed_input_raw = np.append(mixedActivated[l][0], mixedActivated[l][1])
          triple_input_raw = np.append(tripleActivated[l][0], tripleActivated[l][1], tripleActivated[l][2])

        # Output Layer (needs to be sigmoid activation)
        regOutput = self.vectorizedSigmoid(np.dot(reg_input_raw, weights[-1]))
        mixedOutput = self.vectorizedSigmoid(np.dot(mixed_input_raw, mixedWeights[-1]))
        tripleOutput = self.vectorizedSigmoid(np.dot(triple_input_raw, tripleWeights[-1]))

        ########################## CALCULATE ERROR ##########################
        error = (regOutput - sol)
        mixedError = (mixedOutput - sol)
        tripleError = (tripleOutput - sol)

        currentError += (error ** 2)
        currentMixedError += (mixedError ** 2)
        currentTripleError += (tripleError ** 2)

        ########################## BACKPROP ##########################
        reg_delta_output = self.delta_output(error, regOutput)
        mixed_delta_output = self.delta_output(mixedError, mixedOutput)
        triple_delta_output = self.delta_output(tripleError, tripleOutput)

        regDelta, mixedDelta, tripleDelta = [reg_delta_output], [mixed_delta_output], [triple_delta_output]

        for l in range(num_hidden_layers, 0, -1):  
          reg, mixed, triple = self.delta_hidden(regActivated[l-1], mixedActivated[l-1], tripleActivated[l-1], regDelta[0], mixedDelta[0], tripleDelta[0])
          
          # Deltas
          regDelta.insert(0, reg)
          mixedDelta.insert(0, mixed)
          tripleDelta.insert(0, triple)
        
        ########################## UPDATE WEIGHTS ##########################
        for l in range(num_hidden_layers, 0, -1):
          weights[l] -= np.array([alpha * (reg_output_activated[l] * regDelta[l])]).T
          mixedWeights[l] -= np.array([alpha * (mixed_output_activated[l] * mixedDelta[l])]).T
          tripleWeights[l] -= np.array([alpha * (triple_output_activated[l] * tripleDelta[l])]).T
        
        weights[0] -= alpha * np.dot(delta[0], sam.T)
        mixedWeights[0] -= alpha * np.dot(mixedDelta[0], sam.T)
        tripleWeights[0] -= alpha * np.dot(tripleDelta[0], sam.T)

      ########################## CHECKPOINT (output) ##########################
      if j % 10 == 0:
        self.trainErrors += [currentError]
        mixedTrainErrors += [currentMixedError]
        tripleTrainErrors += [currentTripleError]

        tripleValError, mixedValError, valError = 0.0, 0.0, 0.0
        for _x, _y in zip(X_val, y_val):

          valOutput1 = self.vectorizedSigmoid(np.dot(_x, weights0))
          valregOutput = self.vectorizedSigmoid(np.dot(valOutput1, weights1))
          

          valMixedInput0 = np.dot(_x, mixedWeights0)
          valSigmoidActivated = self.vectorizedSigmoid(valMixedInput0[:self.halfway_point])
          valTanhActivated = self.vectorizedTanh(valMixedInput0[self.halfway_point:])

          valMixedOutput1 = np.append(valSigmoidActivated, valTanhActivated)
          valmixed_output = self.vectorizedSigmoid(np.dot(valMixedOutput1, mixedWeights1))

          valTripleInput0 = np.dot(sam, tripleWeights0)

          valTripleSigmoidActivated = self.vectorizedSigmoid(valTripleInput0[:self.thirdway_point])
          valTripleTanhActivated = self.vectorizedSigmoid(valTripleInput0[self.thirdway_point:2*self.thirdway_point])
          valTripleLinearActivated = self.vectorizedLinear(valTripleInput0[2*self.thirdway_point:])
          valTripleOutput1 = np.append(np.append(valTripleSigmoidActivated, valTripleTanhActivated), valTripleLinearActivated)
          valtriple_output = self.vectorizedSigmoid(np.dot(valTripleOutput1, tripleWeights1))


          valError += ((_y - valregOutput) ** 2.0)
          mixedValError += ((_y - valmixed_output) ** 2.0)
          tripleValError += ((_y - valtriple_output) ** 2.0)

        print("Validation error : ", valError, mixedValError, tripleValError)
        self.validationErrors += [valError]
        mixedValidationErrors += [mixedValError]
        tripleValidationErrors += [tripleValError]

        print('Iteration: {0}\tProgress: {1}%\tError: {2}, {3}, {4}'.format(j, int(100*j/max_loops), currentError, currentMixedError, currentTripleError))

    ########################## WRITE TO FILE ##########################
    np.savetxt('tripleWeights0.txt', weights0)
    np.savetxt('tripleWeights1.txt', weights1)

    """
    testOutput1 = sigmoid(np.dot(X_test, weights0))
    testregOutput = sigmoid(np.dot(testOutput1, weights1))
    testErr = np.sum(np.square(y_test - testregOutput))

    print("#### Test Error : ", testErr)
    """

    ########################## PLOT RESULTS ##########################
    if (plot_results):
      create_graph(211, self.trainErrors, mixedTrainErrors, tripleTrainErrors, 'Training')
      create_graph(212, self.validationErrors, mixedValidationErrors, tripleValidationErrors, 'Validation')
    
      plt.show()   

  ########################## HELPER METHODS ##########################
  def initialize_weight_layer(self, input_size, output_size):
    scalingFactor = 2.0
    return scalingFactor * np.random.random( (input_size, output_size) ) - (scalingFactor / 2.0)
 
  def initialize_vectors(self):
    self.vectorizedSigmoid, self.vectorizedTanh, self.vectorizedLinear = np.vectorize(Activation.sigmoid), np.vectorize(Activation.tanh), np.vectorize(Activation.linear)
    self.vectorizedSigmoidPrime, self.vectorizedTanhPrime, self.vectorizedLinearPrime = np.vectorize(Activation.sigmoidPrime), np.vectorize(Activation.tanhPrime), np.vectorize(Activation.linearPrime)

  def activate_reg(self, reg_input):
    return self.vectorizedSigmoid(reg_input)

  def activate_mixed(self, mixed_input):
    sigmoidActivated = self.vectorizedSigmoid(mixed_input[:self.halfway_point])
    tanhActivated = self.vectorizedTanh(mixed_input[self.halfway_point:])
    return [sigmoidActivated, tanhActivated]

  def activate_triple(self, triple_input):
    tripleSigmoidActivated = self.vectorizedSigmoid(triple_input[:self.thirdway_point])
    tripleTanhActivated = self.vectorizedSigmoid(triple_input[self.thirdway_point:2*self.thirdway_point])
    tripleLinearActivated = self.vectorizedLinear(triple_input[2*self.thirdway_point:])
    return [tripleSigmoidActivated, tripleTanhActivated, tripleLinearActivated]

  def delta_output(self, error, output):
    return error * self.vectorizedSigmoidPrime(output)

  def delta_hidden(self, regActivated, mixedActivated, tripleActivated, delta2, mixedDelta2, tripleDelta2):
    # Regular
    regPrime = regActivated

    # Mixed
    mixedPrime = np.append(self.vectorizedSigmoidPrime(mixedActivated[0]), self.vectorizedTanhPrime(mixedActivated[1]))

    # Triple
    tripleSigmoidPrimeDelta = self.vectorizedSigmoidPrime(tripleActivated[0])
    tripleTanhPrimeDelta = self.vectorizedTanhPrime(tripleActivated[1])
    tripleLinearPrimeDelta = self.vectorizedLinearPrime(tripleActivated[2])
    triplePrime = np.append(np.append(tripleSigmoidPrimeDelta, tripleTanhPrimeDelta), tripleLinearPrimeDelta)

    # Resulting deltas
    delta1 = delta2.dot(weights1.T) * Activation.sigmoidPrime(regPrime)
    mixedDelta1 = mixedDelta2.dot(mixedWeights1.T) * mixedPrime
    tripleDelta1 = tripleDelta2.dot(tripleWeights1.T) * triplePrime

    return delta1, mixedDelta1, tripleDelta1


  def create_graph(self, magic, errors, mixedErrors, tripleErrors, title):
      plt.subplot(magic)
      plt.plot(list(range(len(errors))), errors, 'g')
      plt.plot(list(range(len(mixedErrors))), mixedErrors, 'r')
      plt.plot(list(range(len(tripleErrors))), tripleErrors, 'b')
      plt.legend(['Regular', 'Mixed', 'Triple'])
      plt.ylabel("SSE")
      plt.xlabel("Iterations")
      plt.title(title + " errors vs. iteration")


  def predict(self, test_subject):

    hidden_layer_size = self.feature_count
    output_layer_size = 1

    tripleWeights0 = np.loadtxt('tripleWeights0.txt')
    tripleWeights1 = np.loadtxt('tripleWeights1.txt')

    vectorizedSigmoid, vectorizedTanh, vectorizedLinear = np.vectorize(Activation.sigmoid), np.vectorize(Activation.tanh), np.vectorize(Activation.linear)

    valTripleInput0 = np.dot(test_subject, tripleWeights0)
    valTripleSigmoidActivated = vectorizedSigmoid(valTripleInput0[:int(hidden_layer_size / 3.0)])
    valTripleTanhActivated = vectorizedSigmoid(valTripleInput0[int(hidden_layer_size / 3.0):int(2.0 * hidden_layer_size / 3.0)])
    valTripleLinearActivated = vectorizedLinear(valTripleInput0[int(2.0 * hidden_layer_size / 3.0):])
    valTripleOutput1 = np.append(np.append(valTripleSigmoidActivated, valTripleTanhActivated), valTripleLinearActivated)
    valtriple_output = vectorizedSigmoid(np.dot(valTripleOutput1, tripleWeights1))

    return valtriple_output

  