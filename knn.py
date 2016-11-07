import math
import operator 

def euclidean_distance(house1, house2):
	distance = 0
	for i in range(len(house1)):
		distance += pow((house1[i] - house2[i]), 2)
	return math.sqrt(distance)

def get_neighbors(neighborhood, house, k):
	distances = []
	for i in range(len(neighborhood)):
		distance = euclidean_distance(house, neighborhood[i])
		distances.append((neighborhood[i], i, distance))

	distances.sort(key=operator.itemgetter(2))

	neighbors = []
	for i in range(k):
		neighbors.append(distances[i])
		
	return neighbors
