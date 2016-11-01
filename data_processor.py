import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import matplotlib.legend_handler as legend_handler
import random
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

# Suppress warnings we want to ignore
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# pd.options.mode.chained_assignment = None  # default='warn'

index_zipcode = 13
index_lat = 14
index_long = 15

means = []
devs = []

def featureScale(x):
	global means
	global devs
	means = np.mean(x)
	devs = np.std(x)
	return np.array((x - means) / devs)

def invertScale(val):
	# print(means, devs)
	return val * devs + means

# data dictionary to hold all the values we read in, drop month and day
data_full = pd.read_csv('kc_house_data.csv')
data = data_full.drop(['id'], axis=1)
# data = data.drop([''])

# X and y for our learning (as matrices)
X = data.as_matrix(columns=data.columns[1:])
y = data['price'].values


def cross_validation(_X=featureScale(X), _y=featureScale(y)):
	indices = _X[:,index_zipcode].argsort()
	_y = _y[indices]
	_X = _X[indices]
	# print(indices)

	X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []

	for i in range(len(_X)):
		rand = random.random()
		if (rand <= 0.8):
			X_train.append(_X[i])
			y_train.append(_y[i])
		elif (rand > 0.9):
			X_val.append(_X[i])
			y_val.append(_y[i])
		else:
			X_test.append(_X[i])
			y_test.append(_y[i])

	print('Training size: {0}\tValidation size: {1}\tTest size: {2}'.format(len(X_train), len(X_val), len(X_test), len(y_train)))

	# X_train, X_val, y_train, y_val = train_test_split(_X, _y, test_size=0.2, random_state=42)
	# X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

	return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)

def cross_validation_random(_X=featureScale(X), _y=featureScale(y)):
	X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []

	for i in range(len(_X)):
		rand = random.random()
		if (rand <= 0.8):
			X_train.append(_X[i])
			y_train.append(_y[i])
		elif (rand > 0.9):
			X_val.append(_X[i])
			y_val.append(_y[i])
		else:
			X_test.append(_X[i])
			y_test.append(_y[i])

	print('Training size: {0}\tValidation size: {1}\tTest size: {2}'.format(len(X_train), len(X_val), len(X_test)))

	# X_train, X_val, y_train, y_val = train_test_split(_X, _y, test_size=0.2, random_state=42)
	# X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

	return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)

def cross_validation_comparison(_X=featureScale(X), _y=featureScale(y)):
	X_train, X_val, y_train, y_val = train_test_split(_X, _y, test_size=0.2, random_state=42)
	X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

	return X_train, y_train, X_val, y_val, X_test, y_test




