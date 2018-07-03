import random
import numpy as np
from bitstring import BitArray


generations = 10
population_size = 5
selection_size = 4
problem_size = 3
n_random_cells = 2
clone_rate = 0.5
mutation_rate = -2.5

min_x = -5
max_x = 5

min_y = -5
max_y = 5

data = []

def problem( x, y ):
	return x**2 + y**2

def binary_to_x( binary_list ):

	x = BitArray(binary_list[:16])
	y = BitArray(binary_list[16:])
	x = x.uint
	y = y.uint

	x = min_x + (( max_x - min_x ) * x ) / (( 2.**16 - 1. ))
	y = min_y + (( max_y - min_y ) * y ) / (( 2.**16 - 1. ))

	return [x , y]

def create_solution():
	x = list(np.random.choice([0, 1], size=(32,)))
	x, y = binary_to_x( x )
	sol = {"x": x, "y": y, "func": problem(x,y) }
	return sol

def create_population():
	for i in range( population_size ):
		data.append( sol )

def clonalg():
	iteration = 0
	create_population()
	for i in range( n_iterations ):
		print("+++++++ iteration ", i)

if __name__=="__main__":
	print("Population")

	x = create_solution()
	print( x )