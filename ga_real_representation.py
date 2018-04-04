import numpy as np
from bitstring import BitArray
import random
from operator import itemgetter
import math

alpha = 0.5
beta = 0.6

n_decimals = 4

lower_limit = -100
upper_limit = 100

n_population = 5
cromosome_size = 2

iterations = 100
cross_prob = 0.9
mutation_prob = 0.05
k_adversaries = 3

data = []

def fitness(x, y):
	return 0.5 - ( (math.sin(math.sqrt(x**2 + y**2)))**2 )/(1.0 + 0.001*(x**2 + y**2))**2

def get_random_cromosome(lower_limit, upper_limit, n_decimals):
	return round(random.uniform(lower_limit, upper_limit), n_decimals)

def generate_population(n_population, cromosome_size):
	""" Receives as inputs the individuals in a population and cromosome size
    	then generates the population that is saved in the global varible data
    """
	population =  []
	for i in range(0, n_population):
		cromosome_1 = get_random_cromosome(lower_limit, upper_limit, n_decimals)
		cromosome_2 = get_random_cromosome(lower_limit, upper_limit, n_decimals)
		population.append([cromosome_1, cromosome_2])
	print("Generating population:")
	print('\n'.join('  '.join(map(str,i)) for i in population))
	for i in population:
		data.append([i])

generate_population(n_population, cromosome_size)