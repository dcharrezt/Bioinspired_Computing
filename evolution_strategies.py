import math
import random
import numpy as np
from operator import itemgetter
import copy

deviation = 0.3
n_population = 1
individual_size = 4

x_upper_limit = 2.048
x_lower_limit = -2.048

n_iteration = 1000
alpha = 2.

infinite = -1

def initialize_population( n_population):
	print("Initializing individual: ")
	for i in range(n_population):
		tmp = []
		tmp.append(random.uniform(x_lower_limit, x_upper_limit))
		tmp.append(random.uniform(x_lower_limit, x_upper_limit))
		tmp.append(deviation)
		tmp.append(deviation)
		tmp.append(-np.inf)
	print(tmp)
	return tmp


def std_deviation(x, dev):
	return (math.exp(-0.5 * (x / dev) ** 2)) / \
		   (dev * math.sqrt(2 * math.pi))


def integral(lim_inf, lim_sup, dev, delta, rnd):
	area = 0.
	aux_sum = std_deviation(lim_inf, dev)
	aux = std_deviation(lim_inf, dev)

	lin_space = np.arange(lim_inf + delta, lim_sup, delta)
	for i in lin_space:
		aux_sum = std_deviation(i, dev)
		area += (aux + aux_sum)
		if (area * (delta / 2.) ) > rnd:
			return i
		aux = aux_sum
	return -1 * infinite

def mutate(individual):
	lim_inf = -4
	lim_sup = 4
	delta = 0.01

	a = individual[0] + integral(lim_inf, lim_sup, individual[2], \
										delta, random.random())
	b = individual[1] + integral(lim_inf, lim_sup, individual[3], \
										delta, random.random())
	c = individual[2]
	d = individual[3]

	mutated = [a, b, c, d, -np.inf]
	return mutated

def fitness_function(x1, x2):
	return 100*(x1**2 - x2)**2 + (1-x1)**2

def evaluate_population(db):
	db[4] = fitness_function(db[0], db[1])

def check_x_values( x1, x2):
	if (x_lower_limit<=x1<=x_upper_limit) and \
				(x_lower_limit<=x2<=x_upper_limit):
		return True
	return False

def ep_algorithm():
	iteration = 0
	c = 0.817
	n = 0
	successes = 0
	failures = 0
	data = initialize_population( n_population)
	evaluate_population(data)

	for i in range(n_iteration):
		print("\niteration #  ", iteration)

		tmp = copy.deepcopy( data )
		while(True):
			if( tmp == data ):
				print("Equal ##")
			else:
				print("data\t", data)
				print("tmp\t", tmp)

			if(n == 5):
				print("Change sigma")
				print("successes\t", successes)
				print("failures\t", failures)
				ps = successes / ( successes + failures )
				n = 0
				successes = 0
				failures = 0
				if ps < 1/5:
					tmp[2] *= c
					tmp[3] *= c
				elif ps > 1/5:
					tmp[2] /= c
					tmp[3] /= c

			ms_dd = mutate( tmp )
			print("MUTATED\t", ms_dd)

			evaluate_population(ms_dd)

			# print("fms\t", ms_dd[4])
			# print("fdata\t", data[4])
			# print("Check\t", check_x_values(ms_dd[0], ms_dd[1]))
			if( ms_dd[4] >= data[4] and check_x_values(ms_dd[0], ms_dd[1])):
				successes += 1
				n += 1
				data = copy.deepcopy(ms_dd)
				break
			else:
				failures += 1
				n += 1
			# print("n", n)


		print("individual\t", data)
		print("Fitness\t", data[4])

		iteration += 1

	print("individual\t", data)

if __name__ == "__main__":
	ep_algorithm()

