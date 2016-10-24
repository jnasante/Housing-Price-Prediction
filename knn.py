import math
import operator 

def euclidean_distance(house1, house2):
	distance = 0
	for i in range(len(house1)):
		distance += pow((house1[i] - house2[i]), 2)
	return math.sqrt(distance)

def get_neighbors(X_train, house, k):
	distances = []
	for i in range(len(X_train)):
		distance = euclidean_distance(house, X_train[i])
		distances.append((X_train[i], distance))

	distances.sort(key=operator.itemgetter(1))

	neighbors = []
	for i in range(k):
		neighbors.append(distances[i][0])
		
	return neighbors

