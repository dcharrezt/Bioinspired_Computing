import random
import numpy as np
from bitstring import BitArray
import copy
import math

generations = 100
population_size = 4
selection_size = 4
n_random_cells = 2
clone_rate = 0.5
mutation_factor = -2.5
n_clones = int(population_size * clone_rate)

min_x = -5
max_x = 5

min_y = -5
max_y = 5

data = []
best_solution = {"func": np.inf}

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
		if max_f - min_f == 0 :
			i["a"] = 1
		else:
			i["a"] = 1-(i["func"]/( max_f - min_f ))

def create_population():
	for i in range( population_size ):
		data.append( create_solution() )

def clone_and_hypermutation():
	tmp = []
	for i in range( population_size ):
		for j in range( n_clones ):
			clone = copy.deepcopy( data[i] )
			try:
				mutation_rate = math.exp(mutation_factor * clone["a"])
			except OverflowError:
				mutation_rate = 0
			for k in range (len( clone["binary"] )):
				rand = random.random()
				if( rand < mutation_rate ):
					if clone["binary"][k] == 0:
						clone["binary"][k] = 1
					else:
						clone["binary"][k] = 0
			x_tmp, y_tmp = binary_to_x( clone["binary"] )
			clone["x"] = x_tmp
			clone["y"] = y_tmp
			clone["func"] = problem( clone["x"], clone["y"] )
			tmp.append( copy.deepcopy(clone) )
	return tmp

def create_random_cells():
	tmp = []
	for i in range( n_random_cells ):
		tmp.append( create_solution() )
	return tmp

def clonalg():
	global data
	global best_solution
	iteration = 0

	create_population()
	for i in range( generations ):
		print("+++++++ iteration ", i)
		affinity()
		tmp = []
		tmp = clone_and_hypermutation()
		rnd_cells = create_random_cells()
		
		data = data + tmp + rnd_cells
		sorted_data = sorted(data, key=lambda k: k["func"] )
		data = []
		for i in range( population_size ):
			data.append( sorted_data[i] )
#			print(data[i])
		sorted_data = []
		if best_solution["func"] > data[0]["func"]:
			best_solution = copy.deepcopy( data[0] )


if __name__=="__main__":
	clonalg()
	print("++++++++++++++ Best solution ")
	print( best_solution )
