import numpy as np
import random


city_distances = [ [0, 12, 3, 23, 1, 5, 23, 56, 12, 11],
				   [12, 0, 9, 18, 3, 41, 45, 5, 41, 27],
				   [3, 9, 0, 89, 56, 21, 12, 48, 14, 29],
				   [23, 18, 89, 0, 87, 46, 75, 17, 50, 42],
				   [1, 3, 56, 87, 0, 55, 22, 86, 14, 33],
				   [5, 41, 21, 46, 55, 0, 21, 76, 54, 81],
				   [23, 45, 12, 75, 22, 21, 0, 11, 57, 48],
				   [56, 5, 48, 17, 86, 76, 11, 0, 63, 24],
				   [12, 41, 14, 50, 14, 54, 57, 63, 0, 9],
				   [11, 27, 29, 42, 33, 81, 48, 24, 9, 0] ]

pheromone_matrix = []
visibility_matrix = []

first_city = 0

p = 0.99
alpha = 1
beta = 1
Q = 1
initial_pheromones = .1

n_ants = 3
n_iterations = 100
n_cities = 10

cities = [ 'A','B','C','D','E','F','G','H','I','J' ]

def print_matrix( matrix ):
	for i in range( n_cities ):
		if(i==0):
			print(" \tA\tB\tC\tD\tE\tF\tG\tH\tI\tJ" )
		for j in range( n_cities ):
			if(j==0):
				print(cities[i], end='\t')
			print(matrix[i][j], end='\t')
		print()

def create_pheromone_matrix():
	for i in range( n_cities ):
		for j in range( n_cities ):


def as_algorithm():
	iteration = 0

	for i in range( n_iterations ):

		print("Iteration # ", iteration)


if __name__ == "__main__":
	# as_algorithm()
	print_matrix( city_distances )
	# print(cities)