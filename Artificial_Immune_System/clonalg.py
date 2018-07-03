import random
import numpy as np
from bitstring import BitArray


generations = 10
population_size = 4
selection_size = 4
problem_size = 3
n_random_cells = 2
clone_rate = 0.5
mutation_factor = -2.5
n_clones = population_size * clone_rate

min_x = -5
max_x = 5

min_y = -5
max_y = 5

data = []
best_solution = {}

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
	b_list = list(np.random.choice([0, 1], size=(32,)))
	x, y = binary_to_x( b_list )
	sol = {"x": x, "y": y, "func": problem(x,y), "a": np.inf, "binary": b_list }
	return sol

def affinity():
	fs = []
	for i in data:
		fs.append( i["func"] )
	max_f = max( fs )
	min_f = min( fs )

	for i in data:
		i["a"] = 1-a["func"]/( max_f - min_f )

def create_population():
	for i in range( population_size ):
		data.append( sol )

def clone_and_hypermutation():
	print()

def create_random_cells()
	print()

def clonalg():
	iteration = 0

	create_population()
	for i in range( n_iterations ):
		print("+++++++ iteration ", i)
		affinity()
		tmp = []
		tmp = clone_and_hypermutation()
		rnd_cells = create_random_cells()

if __name__=="__main__":
	print("Population")

	x = create_solution()
	print( x )