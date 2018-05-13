import math
import random
import numpy as np

x_lower_limit = -10.
x_upper_limit = 10.
deviation = .3
individual_size = 2
population_size = 5
n_adversaries = 3
data = list()

def generate_individual( size, x_lower_limit, x_upper_limit ):
	return {"fitness": np.inf, "xs":[ random.uniform(x_lower_limit, x_upper_limit) \
						for i in range(size)], "dev": [deviation]*individual_size}

def generate_population( population_size ):
	for x in range( population_size ):
		data.append( generate_individual( individual_size, x_lower_limit, \
															x_upper_limit) )
def evaluate_population():
	for i in data:
		i["fitness"] = fitness_function( i["xs"][0], i["xs"][1] )

def tournament_selection( n_adversaries ):
	adversaries = np.random.permutation( list( range( population_size ) ) )
	tmp = [ data[i] for i in adversaries[:n_adversaries]]
	return min(tmp, key=lambda item: item["fitness"])

def crossover( parent_1, parent_2):
	xs = [ .5*( parent_1["xs"][i] + parent_2["xs"][i] ) \
						for i in range(len(parent_1["xs"])) ]
	devs = [ math.sqrt( parent_1["dev"][i] + parent_2["dev"][i] ) \
						for i in range(len(parent_1["dev"])) ]
	fitness = fitness_function( xs[0], xs[1] )
	return {"xs": xs, "devs": devs, "fitness": fitness}
	
def fitness_function( x_1, x_2):
	return -math.cos(x_1)*math.cos(x_2)*math.exp(-(x_1-math.pi)**2 - \
										(x_2-math.pi)**2)

def valid_individual( individual ):
	if( x_lower_limit <= individual["xs"][0] <= x_upper_limit and \
		x_lower_limit <= individual["xs"][1] <= x_upper_limit ):
		return True
	return False


def mutation( individual ):


def ep_u_1():
	generate_population( population_size )
	evaluate_population()
	print(data)
	tournament_selection( n_adversaries )
	return True


if __name__ == "__main__":
	ep_u_1()
	# print( fitness_function( math.pi, math.pi) )