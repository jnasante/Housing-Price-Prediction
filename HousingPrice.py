from data_processor import X, y, invertScale, cross_validation, cross_validation_comparison
from CustomNeuralNetwork import NeuralNetwork
import numpy as np
from knn import get_neighbors

# Prepare the data
X_train, y_train, X_val, y_val, X_test, y_test = cross_validation()

def get_neural_network(retrain=False):
	# Neural Network
	nn = NeuralNetwork(feature_count=len(X[0]))
	if (retrain):
		nn.train_regression(X_train, y_train.T, X_val, y_val.T, X_test, y_test.T, max_loops=100, plot_results=True)

	return nn

def predict_price(test_subject, nn):
	# K-Nearest Neighbors
	neighbors = get_neighbors(X_test, test_subject, k=10)

	# Combine forces
	predictions_sum = 0.0
	knn_predictions_sum = 0.0
	for neighbor in neighbors:
		predictions_sum += nn.predict(neighbor[0])
		knn_predictions_sum += y_test[neighbor[1]]
	price_combine = predictions_sum/len(neighbors)
	price_knn = knn_predictions_sum/len(neighbors)

	return price_combine, price_knn

	# print('Neural Network Prediction: {0}'.format(nn.predict(test_subject)))
	# print('KNN Prediction: {0}'.format(price))
	# print('Actual: {0}'.format(actual_price))

def square_diff(true, pred):
	return (true-pred)**2

def compare_knn(test_set=X_test):
	print('Comparing error of NN, KNN, & combined')
	nn = get_neural_network()

	knnErrors = []
	nnErrors = []
	errors = []
	for i in range(len(test_set)):
		if (i % 100 == 0):
			print('Iteration: {0}\tProgress: {1}%'.format(i, int(100*i/len(test_set))))

		# Bare Neural Network
		price = nn.predict(X_test[i])
		err = invertScale(square_diff(price, y_test[i]))
		nnErrors.append(err)

		# Neural Net clothed with KNN / Bare KNN
		price, knn_price = predict_price(X_test[i], nn)
		err = invertScale(square_diff(price, y_test[i]))
		errors.append(err)
		knn_err = invertScale(square_diff(knn_price, y_test[i]))
		knnErrors.append(knn_err)

	knnError = np.mean(knnErrors)
	nnError = np.mean(nnErrors)
	error = np.mean(errors)

	print('NN MSE: {0}\tKNN MSE:{1}\ttogether:{2}'.format(nnError, knnError, error))

def test_sample():
	test_subject = 50
	nn = get_neural_network(retrain=True)
	n = nn.predict(X_test[test_subject])
	k = predict_price(X_test[test_subject], nn)
	print('Neural Net: {0}, {1}'.format(n, invertScale(n)))
	print('KNN: {0}, {1}'.format(k[1], invertScale(k[1])))
	print('Combined: {0}, {1}'.format(k[0], invertScale(k[0])))
	print('Actual Price: {0}, {1}:'.format(y_test[test_subject], invertScale(y_test[test_subject])))


# price = predict_price(X_test[0], get_neural_network())
# print(price)

test_sample()
#compare_knn()
