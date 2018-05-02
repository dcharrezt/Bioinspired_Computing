import math
import random
import numpy as np
from operator import itemgetter
import copy

deviation = .3
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
	lim_inf = -5
	lim_sup = 5
	delta = 0.01

	a = individual[0] + integral(lim_inf, lim_sup, individual[2], \
	                                    delta, random.random())
	b = individual[1] + integral(lim_inf, lim_sup, individual[3], \
	                                    delta, random.random())
	c = individual[2]
	d = individual[3]
	return [a, b, c, d, -np.inf]


def fitness_function(x1, x2):
    return 100*(x1**2 - x2)**2 + (1-x1)**2

def evaluate_population(db):

    db[4] = fitness_function(db[0], db[1])


def ep_algorithm():
    iteration = 0
    data = initialize_population( n_population)
    evaluate_population(data)
    c = 0.817

    n = 0
    successes = 0
    failures = 0

    for i in range(n_iteration):
        print("\niteration #  ", iteration)

        tmp = copy.deepcopy( data )
        while(True):
            ms_dd = mutate( tmp )
            if x_lower_limit<=ms_dd[0]<=x_upper_limit and \
                x_lower_limit<=ms_dd[1]<=x_upper_limit:
                break

        evaluate_population(ms_dd)

        if( ms_dd[4] >= data[4]):
            successes += 1
            n += 1
            data = copy.deepcopy(ms_dd)
        elif(ms_dd[4] < data[4] ):
            failures += 1
            n += 1 

        print("individual\t", data)
        print("Fitness\t", data[4])

        if(n == 5):
            ps = successes / ( successes + failures )
            n = 0
            successes = 0
            failures = 0
            if ps < 1/5:
                data[2] *= c
                data[3] *= c
            elif ps > 1/5:
                data[2] /= c
                data[3] /= c
    
        iteration += 1

    print("individual\t", data)
ep_algorithm()
#print (fitness_function(math.pi, math.pi) )
