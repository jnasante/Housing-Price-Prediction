import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import matplotlib.legend_handler as legend_handler
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

# Suppress warnings we want to ignore
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# pd.options.mode.chained_assignment = None  # default='warn'

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

# X and y for our learning (as matrices)
X = data.as_matrix(columns=data.columns[1:])
y = data['price'].values

def cross_validation(_X=featureScale(X), _y=featureScale(y)):
	X_train, X_val, y_train, y_val = train_test_split(_X, _y, test_size=0.2, random_state=42)
	X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

	return X_train, y_train, X_val, y_val, X_test, y_test




