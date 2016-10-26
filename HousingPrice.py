from data_processor import X, y, cross_validation, invertScale
from CustomNeuralNetwork import NeuralNetwork
import numpy as np
from knn import get_neighbors

# Prepare the data
X_train, y_train, X_val, y_val, X_test, y_test = cross_validation()

def get_neural_network(retrain=False):
	# Neural Network
	nn = NeuralNetwork()
	if (retrain):
		nn.train_regression(X_train, y_train.T, X_val, y_val.T, X_test, y_test.T, max_loops=500)

	return nn

def predict_price(test_subject, nn):

	# K-Nearest Neighbors
	neighbors = get_neighbors(X_test, test_subject, 3)

	# Combine forces
	predictions_sum = 0.0
	for neighbor in neighbors:
		predictions_sum += nn.predict(neighbor)
	price = predictions_sum/len(neighbors)

	return price

	# print('Neural Network Prediction: {0}'.format(nn.predict(test_subject)))
	# print('KNN Prediction: {0}'.format(price))
	# print('Actual: {0}'.format(actual_price))

def compare_knn():
	print('Calculating error with KNN')
	nn = get_neural_network()

	testErrors = []
	for i in range(len(X_test)):
		if (i % 100 == 0):
			print('Iteration: {0}'.format(i))
		price = predict_price(X_test[i], nn)
		testErrors.append((price-y_test[i])**2)
	error = np.sum(testErrors)

	print('Test Error: {0}'.format(error))



# price = predict_price(X_test[0], get_neural_network())
# print(price)

test_subject = 0
nn = get_neural_network()
n = nn.predict(X_test[test_subject])
p = predict_price(X_test[test_subject], nn)
print(invertScale(n))
print(invertScale(p))
print(invertScale(y_test[test_subject]))
# print(nn.batch_predict(X_test))
# print()

# compare_knn()
