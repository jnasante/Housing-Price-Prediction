import numpy as np
import tensorflow as tf
import data_processor as data
import math
from random import shuffle
from scipy.special import expit
from matplotlib import pyplot as plt

class TFNeuralNetwork():

  def train_regression(self, X_train, y_train, X_val, y_val, X_test, y_test, max_loops=10000, alpha=0.000001, plot_results=False):
    inputLayerSize = 18
    outputLayerSize = 1
    hiddenLayerSize = 18

    y_train = np.reshape(y_train, (-1, 1))
    printFactor = int(max_loops / 10.0)

    # Make it parallel
    self.sess = tf.InteractiveSession(config=tf.ConfigProto(
    intra_op_parallelism_threads=4)) 

    # Construct input layer, save it to this object for prediction
    self.inputs = tf.placeholder(tf.float32, shape=[None, inputLayerSize])

    # Construct output layer variable which is a single continuous value 
    desired_outputs = tf.placeholder(tf.float32, shape=[None, outputLayerSize])

    # Connect 18 inputs to 18 hidden units
    weights_1 = tf.Variable(tf.truncated_normal([inputLayerSize, hiddenLayerSize]))

    # Construct bias variable 
    biases_1 = tf.Variable(tf.zeros([hiddenLayerSize]))

    # connect 2 inputs to every hidden unit. Add bias
    layer_1_outputs = tf.nn.tanh(tf.matmul(self.inputs, weights_1) + biases_1)

    # connect first hidden units to 2 hidden units in the second hidden layer
    weights_2 = tf.Variable(tf.truncated_normal([hiddenLayerSize, outputLayerSize]))
    # [!] The same of above
    biases_2 = tf.Variable(tf.zeros([hiddenLayerSize]))

    # connect the hidden units to the second hidden layer
    self.logits = tf.nn.tanh(
        tf.matmul(layer_1_outputs, weights_2) + biases_2)

    # [!] The error function chosen is good for a multiclass classification taks, not for a XOR.
    self.error_function = 0.5 * tf.reduce_sum(tf.sub(self.logits, desired_outputs) * tf.sub(self.logits, desired_outputs))

    self.train_step = tf.train.GradientDescentOptimizer(alpha).minimize(self.error_function)

    self.sess.run(tf.initialize_all_variables())
    trainErrors = []

    for i in range(max_loops):
      _, loss = self.sess.run([self.train_step, self.error_function], feed_dict={self.inputs: np.array(X_train), desired_outputs: np.array(y_train)})
      if i % printFactor == 0:
        trainErrors += [loss]
        print(loss)

    #plt.subplot(211)
    plt.plot(list(range(len(trainErrors))), trainErrors, 'g')
    plt.legend(['Tanh'])
    plt.ylabel("SSE")
    plt.xlabel("Iterations")
    plt.title("TF Training errors vs. iteration")
    plt.show()

    #print(sess.run(logits, feed_dict={inputs: np.array([[1.0, 1.0]])}))

  def predict(self, test_subject):
    return self.sess.run(self.logits, feed_dict={self.inputs: np.reshape(test_subject, (1, 18))})

