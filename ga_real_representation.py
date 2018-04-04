import numpy as np
from bitstring import BitArray
import random
from operator import itemgetter
import math

alpha = 0.5
beta = 0.6

n_population = 100
cromosome_size = 7
iterations = 100
cross_prob = 0.9
mutation_prob = 0.05
k_adversaries = 3


def fitness(x, y):
	return 0.5 - ( (math.sin(math.sqrt(x**2 + y**2)))**2 )/(1.0 + 0.001*(x**2 + y**2))**2

ms = fitness(5, 8)

print(ms)