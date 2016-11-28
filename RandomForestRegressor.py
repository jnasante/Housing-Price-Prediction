from data_processor import X, y, invertScale, cross_validation_comparison
import matplotlib.pyplot as plot
import matplotlib.legend_handler as legend_handler
import pandas as pd
import numpy as np
import math
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, auc

should_graph = True

# Constants/global variables
X_train, X_validation, X_test = [], [], []
y_train, y_validation, y_test = [], [], []
clf = None

# ------------------------------------------------------------------------------ #
# ----------------------------------- Setup ------------------------------------ #
# ------------------------------------------------------------------------------ #

def graph(points, title='', ylabel=''):
	if (should_graph == False):
		return

	fig, axes = plot.subplots(nrows=2, ncols=1, sharey=False, sharex=False)

	axes[0].plot(range(1, len(points[0])+1), points[0])
	axes[0].set_title(title)
	axes[0].set_xlabel('Max Depth')
	axes[0].set_ylabel(ylabel)

	axes[1].plot(range(1, len(points[1])+1), points[1])
	axes[1].set_title(title)
	axes[1].set_xlabel('Max Features')
	axes[1].set_ylabel(ylabel)

	plot.show()

def initialize():
	global X_train
	global X_validation
	global X_test
	global y_train
	global y_validation
	global y_test

	# Split data into train/validate/test
	X_train, y_train, X_validation, y_validation, X_test, y_test = cross_validation_comparison(X, y)

# ------------------------------------------------------------------------------ #
# ---------------------------- Performance Metrics ----------------------------- #
# ------------------------------------------------------------------------------ #

def get_accuracy(y_pred):
	return accuracy_score(y_test, y_pred)

# ------------------------------------------------------------------------------ #
# --------------------------------- Regression --------------------------------- #
# ------------------------------------------------------------------------------ #

def train_regressor():
	models = []
	errors = []
	tries = 10

	for i in range(tries):
		clf = RandomForestRegressor()

		model = clf.fit(X_train, y_train)
		models.append(model)
		y_pred = clf.predict(X_validation)
		errors.append(mean_squared_error(y_validation, y_pred))

	return models[errors.index(min(errors))]

def random_forest_regressor():
	global clf

	title = 'Random Forest Regressor'
	print(title)

	clf = train_regressor()
	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_validation)
	print('\tMSE: {0}'.format(mean_squared_error(y_validation, y_pred)))

def regression():
	random_forest_regressor()

def predict():
	prediction = clf.predict(X_test[0].reshape(1, -1))
	print('Prediction: ${0}\tActual: ${1}'.format(round(prediction[0], 2), round(y_test[0], 2)))


# ------------------------------------------------------------------------------ #
# -------------------------------- Main Program -------------------------------- #
# ------------------------------------------------------------------------------ #

# Setup
print('Initializing SkyNet...\n')
initialize()

# Regression
regression()

# Predict
predict()

print('\nSkyNet is now self-aware.')
