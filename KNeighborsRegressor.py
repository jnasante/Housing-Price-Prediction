from data_processor import X, y, invertScale, cross_validation_comparison
import matplotlib.pyplot as plot
import matplotlib.legend_handler as legend_handler
import pandas as pd
import random
import numpy as np
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Other constants
# ks = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 ]
ks = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
bestModel, bestError = 0, 0.0

# Error for all splits
training_error = []
validation_error = []
testing_error = []
num_splits = 1

# Split class to hold all splits with models for each split
class Split:
	def __init__(self):
		# Splits to hold data
		self.X_training = []
		self.y_training = []
		self.X_validation = []
		self.y_validation = []
		self.X_testing = []
		self.y_testing = []

		# Prediction after running model
		self.models = []

splits = []
training_error = []
validation_error = []
testing_error = []

def createModels():
	for i in range(0, num_splits):
		print('Creating model')
		split = Split()
		
		# Split data into training/validation/testing
		split.X_training, split.y_training, split.X_validation, split.y_validation, split.X_testing, split.y_testing = cross_validation_comparison(X, y)

		# Knn
		for k in ks:
			split.models.append(KNeighborsRegressor(n_neighbors=k))

		splits.append(split)

def findBestModel():
	global bestModel
	global bestError

	bestModel = 0
	bestError = validation_error[bestModel]
	for k in range(1, len(ks)):
		currValue = validation_error[k]
		if (bestError > currValue):
			bestError = currValue
			bestModel = k

def learn():
	for m in range(len(ks)):
		print('learning for k={0}'.format(ks[m]))
		training_total = 0
		validation_total = 0
		testing_total = 0
		for split in splits:
			# Let's get fit!
			learnedModel = split.models[m].fit(split.X_training, split.y_training)

			# Training
			prediction = learnedModel.predict(split.X_training)
			training_total = training_total + mean_squared_error(split.y_training, prediction)

			# Validation
			prediction = learnedModel.predict(split.X_validation)
			validation_total = validation_total + mean_squared_error(split.y_validation, prediction)

			# Testing
			prediction = learnedModel.predict(split.X_testing)
			testing_total = testing_total + mean_squared_error(split.y_testing, prediction)

		training_error.append(training_total/num_splits)
		validation_error.append(validation_total/num_splits)
		testing_error.append(testing_total/num_splits)

	findBestModel()

def print_results():
	# Find F* (best model - lowest mse after learning)
	print('Best k: ' + str(ks[bestModel]))
	print('Final test MSE: ' + str(testing_error[bestModel]))

def graph_results():
	# Graph everything
	fig, axes = plot.subplots(nrows=3, ncols=1, sharey=False, sharex=False)

	axes[0].plot(range(1, len(training_error)+1), training_error)
	axes[0].set_title("Mean Squared Error (Training)")

	axes[1].plot(range(1, len(validation_error)+1), validation_error)
	axes[1].set_title("Mean Squared Error (Validation)")

	axes[2].plot(range(1, len(testing_error)+1), testing_error)
	axes[2].set_title("Mean Squared Error (Testing)")

	plot.show()

def train_model():
	# Perform logic to split, create models, and learn for each model
	createModels()
	learn()

def predict():
	prediction = splits[0].models[bestModel].predict(splits[0].X_testing[0].reshape(1, -1))
	print('Prediction: {0}\tActual: {1}'.format(prediction[0], splits[0].y_testing[0]))


train_model()
print_results()
graph_results()
predict()



