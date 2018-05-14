import math
import random
import numpy as np

x_lower_limit = -10.
x_upper_limit = 10.
deviation = .3
individual_size = 2
population_size = 10
n_adversaries = 3
n_iterations = 200

infinite = -1.
lim_inf = -4
lim_sup = 4
delta = 0.01

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

	xs = list([ .5*( parent_1["xs"][i] + parent_2["xs"][i] ) \
						for i in range(len(parent_1["xs"])) ])
	devs = list([ math.sqrt( parent_1["dev"][i] + parent_2["dev"][i] ) \
						for i in range(len(parent_1["dev"])) ])
	fitn = fitness_function( xs[0], xs[1] )
	return {"xs": xs, "dev": devs, "fitness": fitn}
	
def fitness_function( x_1, x_2):
	return -math.cos(x_1)*math.cos(x_2)*math.exp(-(x_1-math.pi)**2 - \
										(x_2-math.pi)**2)

def valid_individual( individual ):
	if( x_lower_limit <= individual["xs"][0] <= x_upper_limit and \
		x_lower_limit <= individual["xs"][1] <= x_upper_limit ):
		return True
	return False

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

def mutation( individual ):
	for i in range( len(individual["xs"]) ):
		individual["dev"][i] *= math.exp( integral(lim_inf, lim_sup, \
								individual["dev"][i], delta, random.random() ))
		individual["xs"][i] += integral(lim_inf, lim_sup, \
								individual["dev"][i], delta, random.random() )
	individual["fitness"] = fitness_function( individual["xs"][0], \
													individual["xs"][1])

def ep_u_1():
	global data
	iteration = 0
	generate_population( population_size )
	evaluate_population()

	while( iteration < n_iterations ):
		print("iteration #", iteration)

		while( True ):
			new_individual = crossover( tournament_selection( n_adversaries ), \
										tournament_selection( n_adversaries ) )
			mutation(new_individual)
			if( valid_individual(new_individual) ):
				break

		data.append( new_individual )
		data = sorted( data, key=lambda x: x["fitness"] )
		del data[ population_size ]

		iteration+=1

		for i in range( population_size ):
			print(data[i]["fitness"])
	print(data)

	return True

def ep_u_lambda():
	global data
	iteration = 0
	lambda_size = int(population_size/2)

	generate_population( population_size )
	evaluate_population()

	while( iteration < n_iterations ):
		print("iteration #", iteration)

		for ms in range( lambda_size ):
			while( True ):
				new_individual = crossover( tournament_selection( n_adversaries ), \
											tournament_selection( n_adversaries ) )
				mutation(new_individual)
				if( valid_individual(new_individual) ):
					break
			data.append( new_individual )

		data = sorted( data, key=lambda x: x["fitness"] )

		for ms in range( lambda_size ):
			del data[ len(data) -1 ]

		for i in range( population_size ):
			print(data[i]["fitness"])

		iteration+=1
	print(data)

def ep_u_plus_lambda():
	global data
	iteration = 0
	lambda_size = int(population_size*1.2)

	generate_population( population_size )
	evaluate_population()

	while( iteration < n_iterations ):
		print("iteration #", iteration)

		for ms in range( lambda_size ):
			while( True ):
				new_individual = crossover( tournament_selection( n_adversaries ), \
											tournament_selection( n_adversaries ) )
				mutation(new_individual)
				if( valid_individual(new_individual) ):
					break
			data.append( new_individual )

		data = sorted( data, key=lambda x: x["fitness"] )

		for ms in range( lambda_size ):
			del data[ len(data) -1 ]

		for i in range( population_size ):
			print(data[i]["fitness"])

		iteration+=1
	print(data)

if __name__ == "__main__":
	ep_u_plus_lambda()
	# ep_u_lambda()
	# ep_u_1()
	# print( fitness_function( 4.1180866324905363, 2.9631102046257594) )