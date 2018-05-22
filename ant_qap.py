import random
import numpy as np

distance_matrix = [ [0, 12, 6, 4],
					[12, 0, 6, 8],
					[6, 6, 0, 7],
					[4, 8, 7, 0] ]

flow_matrix = [ [0, 3, 8, 3],
				[3, 0, 2, 4],
				[8, 2, 0, 5],
				[3, 4, 5, 0] ]

n_ants = 4
n_units = 4
initial_pheromones = 1.

alpha = 1
beta = 1

min_pheromone = 0.1
max_pheromone = 1.

n_iterations = 10

pheromone_matrix = np.zeros(( n_cities, n_cities ))
visibility_matrix = np.zeros(( n_cities, n_cities ))

def get_random_ant( n_units ):
	return np.random.permutation( n_unitsn )

def initialize_pheromone_matrix():
	for i in range( n_cities ):
		for j in range( n_cities ):
			if(i!=j):
				pheromone_matrix[i][j] = initial_pheromones

def initialize_visibility_matrix():
	for i in range( n_cities ):
		for j in range( n_cities ):
			if(i!=j):
				visibility_matrix[i][j] = 1.0 / \
								( distance_matrix[i][j] * flow_matrix[i][j] )

def send_ants( n_ants ):

	ants = [ get_random_ant(n_units) for i in range(n_ants) ]
	for ant in ants:
		print("Ant # ", ant, end='')
		for unit in ant:
			print(unit, end='')

def min_max_algorithm():

	initialize_visibility_matrix()
	initialize_pheromone_matrix()

	# for i in range( n_iterations ):
	# 	print

if __name__ == "__main__":

	min_max_algorithm()