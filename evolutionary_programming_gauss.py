import math
import random
import numpy as np
from operator import itemgetter

deviation = .3
n_population = 50
individual_size = 4

x_upper_limit = 10.
x_lower_limit = -10.

n_iteration = 20
alpha = 2.

infinite = -1


def initialize_population(data, n_population):
    for i in range(n_population):
        tmp = []
        tmp.append(random.randint(x_lower_limit, x_upper_limit))
        tmp.append(random.randint(x_lower_limit, x_upper_limit))
        tmp.append(deviation)
        
        tmp.append(deviation)
        data.append([tmp, np.inf])


def std_deviation(x, dev):
    return (math.exp(-0.5 * (x / dev) ** 2)) / \
           (dev * math.sqrt(2 * math.pi))


def integral(lim_inf, lim_sup, dev, delta, rnd):
    area = 0.
    aux_sum = std_deviation(lim_inf, dev)
    aux = std_deviation(lim_inf, dev)

    lin_space = np.arange(lim_inf + delta, lim_sup, delta)
#    print( "asd" )
#    print( lin_space )
#    print( "as2 " )
    for i in lin_space:
        aux_sum = std_deviation(i, dev)
        area += (aux + aux_sum)
        if (area * (delta / 2.) ) > rnd:
            return i
        aux = aux_sum
    return -1 * infinite

def mutate(individual):
	lim_inf = -10
	lim_sup = 10
	delta = 0.01
	
	a = float(individual[0][2] * (1+alpha*integral(lim_inf, lim_sup, deviation, delta , \
															random.random())))
	b = float(individual[0][3] * (1+alpha*integral(lim_inf, lim_sup, deviation, delta , \
															random.random())))

	c = float(individual[0][0] + individual[0][2]*integral(lim_inf, lim_sup, deviation, delta , \
															random.random()))
	d = float(individual[0][1] + individual[0][3]*integral(lim_inf, lim_sup, deviation, delta , \
															random.random()))
	
	return [c, d, a ,b]



def fitness_function(x1, x2):
    return -math.cos(x1) * math.cos(x2) * math.exp(-(x1 - math.pi) ** 2 - (x2 - math.pi) ** 2)


def evaluate_population(db):
    for i in db:
        i[1] = fitness_function(i[0][0], i[0][1])


def ep_algorithm():
    data = []
    iteration = 0
    initialize_population(data, n_population)
    evaluate_population(data)

    while (True):
        print("\niteration #", iteration)

        offspring = []
        for i in data:
            while(True):
                ms_dd = mutate( i )
                if( (-10<=ms_dd[0]<=10) and (-10<=ms_dd[1]<=10 ) ):
                    offspring.append( [ms_dd, np.inf] )
                    break
        evaluate_population(offspring)


        survivors = []
        tmp_1 = []
        tmp_2 = []

        for i, x in zip(data, range(len(data))):
            tmp_1.append([x, data[x][1]])
            tmp_2.append([x, offspring[x][1]])
        tmp_1 = sorted(tmp_1, key=itemgetter(1), reverse=False)
        tmp_2 = sorted(tmp_2, key=itemgetter(1), reverse=False)

        for i in range(int(n_population / 2)):
            survivors.append(data[tmp_1[i][0]])
        for i in range(int(n_population / 2)):
            survivors.append(offspring[tmp_2[i][0]])

        if (iteration >= n_iteration):
            break
        iteration += 1

        data = survivors[:]

    print("RESULTS: ")
    survivors = []
    for i, x in zip(data, range(len(data))):
        tmp_1.append([x, data[x][1]])
    tmp_1 = sorted(tmp_1, key=itemgetter(1), reverse=False)

    for i in range(len(data)):
        survivors.append(data[tmp_1[i][0]])
    for i in survivors:
        print(i,"\n\n")

ep_algorithm()
#print (fitness_function(math.pi, math.pi) )
