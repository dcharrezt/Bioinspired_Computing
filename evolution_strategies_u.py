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
									for i in range(size)], "dev": deviation}

def generate_population( population_size ):
	for x in range( population_size ):
		data.append( generate_individual( individual_size, x_lower_limit, \
															x_upper_limit) )

# def tournament_selection( n_adversaries ):


def fitness_function( x_1, x_2):
	return -cos(x_1)*cos(x_2)*math.exp(-(x_1-math.pi)**2 - \
										(x_2-math.pi)**2)

def ep_u_1():
	generate_population( population_size )
	print(data)
	return True


if __name__ == "__main__":
	ep_u_1()