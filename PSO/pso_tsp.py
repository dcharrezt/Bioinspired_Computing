
import numpy as np
import random

# Parameters 

distances = [ [0, 1, 3, 4, 5],
			  [1, 0, 1, 4, 8],
			  [3, 1, 0, 5, 1],
			  [4, 4, 5, 0, 2],
			  [5, 8, 1, 2, 0] ]

n_iterations = 4
n_particles = 4
n_cities = 5

phi_1 = 1.0
phi_2 = 1.0

data = []
cities = [ 'A', 'B', 'C', 'D', 'E' ]

def create_particle():
	return np.random.permutation( n_cities )

def function_fitness( path );
	cost = 0.
	for i in range( n_cities-1 ):
		cost += distances[ path[i] ][ path[i+1] ]
	return cost
	
def pso():
	print("****** Starting PSO")


if __name__=="__main__":
	pso()