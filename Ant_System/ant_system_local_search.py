import numpy as np
import random


distance_matrix = [ [0, 12, 3, 23, 1, 5, 23, 56, 12, 11, 89, 97, 52], 
					[12, 0, 9, 18, 3, 41, 45, 5, 41, 27, 16, 76, 56], 
					[3, 9, 0, 89, 56, 21, 12, 48, 14, 29, 5, 91, 8],
					[23, 18, 89, 0, 87, 46, 75, 17, 50, 42, 100, 70, 15], 
					[1, 3, 56, 87, 0, 55, 22, 86, 14, 33, 31, 84, 21],
					[5, 41, 21, 46, 55, 0, 21, 76, 54, 81, 92, 37, 22],
					[23, 45, 12, 75, 22, 21, 0, 11, 57, 48, 39, 59, 22],
					[56, 5, 48, 17, 86, 76, 11, 0, 63, 24, 55, 58, 98],
					[12, 41, 14, 50, 14, 54, 57, 63, 0, 9, 44, 18, 52],
					[11, 27, 29, 42, 33, 81, 48, 24, 9, 0, 64, 65, 82],
					[89, 16, 5, 100, 31, 92, 39, 55, 44, 64, 0, 9, 70],
					[97, 76, 91, 70, 84, 37, 59, 58, 18, 65, 9, 0, 50],
					[52, 56, 8, 15, 21, 22, 22, 98, 52, 82, 70, 50, 0] ]

# Parameters
first_city = 3
p = 0.99
alpha = 1
beta = 1
Q = 1
initial_pheromones = .1

n_ants = 3
n_iterations = 100
n_cities = 13

pheromone_matrix = np.zeros(( n_cities, n_cities ))
visibility_matrix = np.zeros(( n_cities, n_cities ))

cities = [ 'A','B','C','D','E','F','G','H','I','J','K','L','M' ]

def print_matrix( matrix, text ):
	print( text )
	for i in range( n_cities ):
		if(i==0):
			print("\tA\tB\tC\tD\tE\tF\tG\tH\tI\tJ\tK\tL\tM" )
		for j in range( n_cities ):
			if(j==0):
				print(cities[i], end='\t')
			print( "{:.3f}".format(matrix[i][j]), end='\t' )
		print()

def initialize_pheromone_matrix():
	for i in range( n_cities ):
		for j in range( n_cities ):
			if(i!=j):
				pheromone_matrix[i][j] = initial_pheromones

def initialize_visibility_matrix():
	for i in range( n_cities ):
		for j in range( n_cities ):
			if(i!=j):
				visibility_matrix[i][j] = 1.0 / distance_matrix[i][j]

def as_algorithm():
	initialize_pheromone_matrix()
	initialize_visibility_matrix()

	print_matrix( distance_matrix, " Distance Matrix " )
	print_matrix( pheromone_matrix, " Pheromone Matrix" )
	print_matrix( visibility_matrix, "Visibility Matrix" )

	# for i in range( n_iterations ):
	# 	print("Iteration # ", i)
	# 	if(i == 0):
	# 		print_matrix( distance_matrix, " Distance Matrix " )
	# 		print_matrix( pheromone_matrix, " Pheromone Matrix" )
	# 		print_matrix( visibility_matrix, "Visibility Matrix" )
	# 	path_list = send_ants()
	# 	cost_list = print_ant_results( path_list )
	# 	update_pheromone_matrix( path_list, cost_list )

	# print_matrix( pheromone_matrix, " Updated Pheromone Matrix ")


if __name__ == "__main__":
	as_algorithm()