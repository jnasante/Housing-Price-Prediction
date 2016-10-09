class Layer():

  def sigmoid(x):
    return 1 / (1 + np.exp(-x))

  def init(self, activationFunction = sigmoid, dim=[0,0]):
    self.weights = np.random(dim)
    self.act = np.vectorize(activationFunction)

  def apply(self, x):
    return self.act(np.dot(self.weights, x))

  def error(self, x=None, y=None):
    if not (x or y):
      return None
    else:
      return x.dot(y)


  def update(self):
    self.weights += 





class NeuralNetwork():
  
  

  def __init__(self, alpha=1E-5, weights=[], w_0=0.0, decay_rate=-1):
    self.alpha = alpha
    self.weights = weights
    self.w_0 = w_0
    self.decay_rate = decay_rate
    
  def predict_regression(self, samples):
    return np.dot(samples, self.weights) + self.w_0

  def train_regression(self, samples, solutions, max_loops=50000):

    
        
    X = np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]])
                    
    y = np.array([[0],
          [1],
          [1],
          [0]])

    np.random.seed(1)

    # randomly initialize our weights with mean 0
    syn0 = 2*np.random.random((3,4)) - 1
    syn1 = 2*np.random.random((4,1)) - 1

    for j in xrange(60000):

      previousOutput = X
      outputs = []

      for layer in layers:
        currentOutput = layer.activate(previousOutput)
        outputs += [currentOutput]
        previousOutput = currentOutput

        """
        # Feed forward through layers 0, 1, and 2
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        """

      # how much did we miss the target value?
      previousActual = y
      previousDelta = error
      for layer in layers[::-1]:
        currentDelta = layer.delta(actual=previousActual)
        
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)
        
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layerDelta = layer.delta(previousDelta * dot(layer.weights) * nonlin(l1,deriv=True))

        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    err = -1
    i = 0
    output = []
  
    # construct initial weight matrix if needed
    if (self.weights == []):
      self.weights = np.random.randn(len(samples[0]))
      self.w_0 = np.random.randn()
    
    for i in range(max_loops):
      # shuffle samples
      samples, solutions = shuffle_lists(samples, solutions)

      if self.decay_rate > 0 and i % self.decay_rate == 0:
        self.alpha = self.alpha / 2.0
    
      for sample, y in zip(samples, solutions):
        prediction = np.dot(sample, self.weights) + self.w_0
        error = self.error(y, prediction)
        delta = self.alpha * error
        self.weights = np.add(np.multiply(delta, sample), self.weights)
        self.w_0 += delta
        err += sse(y, prediction)
        
      err /= len(samples)
      output.append(err)
      
      print "i=", i, ", err=", err#, ", v_err=", v_err
        
    return output
        
  def error(self, y, y_hat):  
    return (y - y_hat)

def sse(y, y_hat):
  return 0.5 * (y - y_hat) ** 2

def shuffle_lists(a, b):
  _a = []
  _b = []
  idx = range(len(a))
  shuffle(idx)
  for i in idx:
    _a.append(a[i])
    _b.append(b[i])
    
  return _a, _b