import math
import random
import numpy as np
from operator import itemgetter



deviation = .3
n_population = 50
individual_size = 4

x_upper_limit = 10.
x_lower_limit = -10.

n_iteration = 1000


def initialize_population( data, n_population ):
	for i in range( n_population ):
		tmp = []
		tmp.append( random.randint( x_lower_limit, x_upper_limit) )
		tmp.append( random.randint( x_lower_limit, x_upper_limit) )
		tmp.append( deviation )
		tmp.append( deviation )
		data.append( [tmp, np.inf] )

def std_deviation( x, dev ):
	return ( math.exp( -0.5 * ( x /  dev )**2 ) ) / \
			( dev * math.sqrt( 2 * math.pi ) )

def integral( lim_inf, lim_sup, dev, delta, rnd ):
	area = 0.
	aux = std_deviation( lim_inf, dev )

	lin_space = np.arange(lim_inf + delta, lim_sup, delta)
	for i in lin_space:
		aux_sum = std_deviation(i, dev)
		area += ( aux + aux_sum )
		if( area * ( delta / 2 ) > rnd ):
			return i

		aux = aux_sum

	return -1* 10e-10

def mutate( individual ):
	tmp = individual[:]
	tmp[0][0] = integral( -1e-10, tmp[0][0], tmp[0][2], \
								0.1, random.random() )
	tmp[0][1] = integral( -1e-10, tmp[0][1], tmp[0][3], \
								0.1, random.random() )

	return tmp

def fitness_function( x1 , x2 ):
	return -math.cos(x1)*math.cos(x2)*math.exp( -(x1-math.pi)**2 - (x2 - math.pi)**2 )

def evaluate_population( db ):
	for i in db:
		i[1] = fitness_function( i[0][0], i[0][1] )

def ep_algorithm():
	data = []
	iteration = 0
	initialize_population( data, n_population )
	evaluate_population( data )

	while( True ):
		print("\niteration #", iteration)

		offspring = []
		for i in data :
			offspring.append(mutate( i ))
		evaluate_population( offspring )

		survivors = []
		tmp_1 = []
		tmp_2 = []

		for i,x in zip(data,range(len(data))):
			tmp_1.append([x, data[x][1] ] )
			tmp_2.append([x, offspring[x][1] ])
		tmp_1 = sorted(tmp_1, key=itemgetter(1), reverse=False)
		tmp_2 = sorted(tmp_2, key=itemgetter(1), reverse=False)

		print("data: ", tmp_1)
		print("Mutated ", tmp_2)

		for i in range( int(n_population / 2) ):
			survivors.append( data[tmp_1[i][0]] )
		for i in range( int(n_population / 2) ):
			survivors.append( offspring[tmp_2[i][0]] )

		if(iteration >= n_iteration):
			break
		iteration +=1 

	print(data)

ep_algorithm()