import numpy as np
import random
import copy
import matplotlib.pyplot as plt

x_lower_limit = 0.
x_upper_limit = 5.

y_lower_limit = 0.
y_upper_limit = 3.

data = []

n_functions = 2
n_iterations = 250
population_size = 50
offspring_size = 10
n_adversaries = 8
beta = 0.5
alpha = 1.
mutation_prob = 0.5   # .0 - 1.

city_distances = [ [0, 12, 3, 23, 1, 5, 23, 56, 12, 11],
				   [12, 0, 9, 18, 3, 41, 45, 5, 41, 27],
				   [3, 9, 0, 89, 56, 21, 12, 48, 14, 29],
				   [23, 18, 89, 0, 87, 46, 75, 17, 50, 42],
				   [1, 3, 56, 87, 0, 55, 22, 86, 14, 33],
				   [5, 41, 21, 46, 55, 0, 21, 76, 54, 81],
				   [23, 45, 12, 75, 22, 21, 0, 11, 57, 48],
				   [56, 5, 48, 17, 86, 76, 11, 0, 63, 24],
				   [12, 41, 14, 50, 14, 54, 57, 63, 0, 9],
				   [11, 27, 29, 42, 33, 81, 48, 24, 9, 0] ]

cost_between_cities = [ [0, 22, 47, 15, 63, 21, 23, 16, 11, 9], 
						[22, 0, 18, 62, 41, 52, 13, 11, 26, 43],
						[47, 18, 0, 32, 57, 44, 62, 20, 8, 36],
						[15, 62, 32, 0, 62, 45, 75, 63, 14, 12],
						[63, 41, 57, 62, 0, 9, 99, 42, 56, 23],
						[21, 52, 44, 45, 9, 0, 77 ,58, 22, 14],
						[23, 13, 62, 75, 99, 77, 0, 30, 25, 60],
						[16, 11, 20, 63, 42, 58, 30, 0, 66, 85],
						[11, 26, 8, 14, 56, 22, 25, 66, 0, 54],
						[9, 43, 36, 12, 23, 14, 60, 85, 54, 0]]


n_cities = len( city_distances )

def generate_individual_tsp():
	return { "cm": np.random.permutation( n_cities ), "f_distance": np.inf, \
					"f_cost": np.inf }

def generate_population_tsp():
	for i in range( population_size ):
		data.append( generate_individual_tsp() )

def evaluate_population_tsp():
	for i in data:
		i["f_distance"] = fitness_distance( i )
		i["f_cost"] = fitness_cost( i )

def fitness_distance( individual ):
	total_distance = 0
	for i in range( n_cities -1 ):
		total_distance += city_distances[ individual["cm"][i] ]\
										[ individual["cm"][i+1] ]
	return total_distance

def fitness_cost( individual ):
	total_cost = 0
	for i in range( n_cities -1 ):
		total_cost += cost_between_cities[ individual["cm"][i] ]\
										[ individual["cm"][i+1] ]
	return total_cost

def PBX_crossover(individual_1, individual_2):
	offspring = []

	mother_cromosome = individual_1["cm"]
	father_cromosome = individual_2["cm"]

	son_1, son_2 = [], []

	mask = np.random.randint(2, size=len(mother_cromosome))

	for i, j in zip(range(len(mother_cromosome)), mask):
	
		if(j == 1):
			son_1.append(father_cromosome[i])
			son_2.append(mother_cromosome[i])
		else:
			son_1.append(-1)
			son_2.append(-1)

	# print(son_1)
	# print(son_2)

	for i in mother_cromosome:
		if (i not in son_1):
			tmp = son_1.index(-1)
			son_1[tmp] = i

	for i in father_cromosome:
		if (i not in son_2):
			tmp = son_2.index(-1)
			son_2[tmp] = i

	# print(*son_1, sep='')
	# print(*son_2, sep='')

	new_individual_1 = { "cm": son_1 }
	new_individual_2 = { "cm": son_2 }

	new_individual_1["f_distance"] = fitness_distance( new_individual_1 )
	new_individual_2["f_distance"] = fitness_distance( new_individual_2 )

	new_individual_1["f_cost"] = fitness_cost( new_individual_1 )
	new_individual_2["f_cost"] = fitness_cost( new_individual_2 )

	offspring.append(new_individual_1)
	offspring.append(new_individual_2)

	return offspring

def mutation_tsp( individual_1 ):
	perm_tmp = list(np.random.permutation( n_cities ))
	individual_1["cm"][perm_tmp[1]], individual_1["cm"][perm_tmp[0]] = \
	 	individual_1["cm"][perm_tmp[0]], individual_1["cm"][perm_tmp[1]]

def tournament_selection_tsp():
	adversaries = np.random.permutation( list( range( population_size ) ) )
	tmp = [ data[i] for i in adversaries[:n_adversaries]]
	sums = [ i["f_distance"]+i["f_cost"] for i in tmp ]
	m_min = min( sums )
	return data[adversaries[sums.index(m_min)]] 

def function_1( x, y):
	return 4*(x**2) + 4*(y**2)

def function_2( x, y):
	return (x-5)**2 + (y-5)**2

def generate_individual():
	return {"x": random.uniform(x_lower_limit, x_upper_limit), \
			"y": random.uniform(y_lower_limit, y_upper_limit), \
			"fitness_1": np.inf, \
			"fitness_2": np.inf }

def generate_population():
	for i in range( population_size ):
		data.append( generate_individual() )

def evaluate_population():
	for i in data:
		i["fitness_1"] = function_1( i["x"], i["y"] )
		i["fitness_2"] = function_2( i["x"], i["y"] )

def tournament_selection():
	adversaries = np.random.permutation( list( range( population_size ) ) )
	tmp = [ data[i] for i in adversaries[:n_adversaries]]
	sums = [ i["fitness_1"]+i["fitness_2"] for i in tmp ]
	m_min = min( sums )
	return data[adversaries[sums.index(m_min)]] 

def BLX_crossover( parent_1, parent_2 ):
	m_beta = random.uniform( beta - alpha, beta + alpha )
	
	m_x = parent_1["x"] + m_beta*( parent_2["x"] - parent_1["x"] )
	m_y = parent_1["y"] + m_beta*( parent_2["y"] - parent_1["y"] )
	m_1 = function_1( m_x, m_y )
	m_2 = function_2( m_x, m_y )

	return {"x": m_x, "y": m_y, "fitness_1": m_1, "fitness_2": m_2 }

def uniform_mutation( individual ):
	if( random.randint(0, 1) ):
		individual["x"] = random.uniform(x_lower_limit, x_upper_limit)
	else:
		individual["y"] = random.uniform(y_lower_limit, y_upper_limit)
	individual["fitness_1"] = function_1( individual["x"], individual["y"] )
	individual["fitness_2"] = function_2( individual["x"], individual["y"] )

def valid_individual( individual ):
	if( x_lower_limit <= individual["x"] <= x_upper_limit and \
		y_lower_limit <= individual["y"] <= y_upper_limit ):
		return True
	return False

def dominate( individual_1, individual_2 ):
		if( ( individual_1["fitness_1"] < individual_2["fitness_1"] and \
			  individual_1["fitness_2"] < individual_2["fitness_2"] ) or \
			( individual_1["fitness_1"] <= individual_2["fitness_1"] and \
			  individual_1["fitness_2"] < individual_2["fitness_2"] ) or \
			( individual_1["fitness_1"] < individual_2["fitness_1"] and \
			  individual_1["fitness_2"] <= individual_2["fitness_2"] ) ):
			return True
		return False 

def dominate_tsp( individual_1, individual_2 ):
		if( ( individual_1["f_distance"] < individual_2["f_distance"] and \
			  individual_1["f_cost"] < individual_2["f_cost"] ) or \
			( individual_1["f_distance"] <= individual_2["f_distance"] and \
			  individual_1["f_cost"] < individual_2["f_cost"] ) or \
			( individual_1["f_distance"] < individual_2["f_distance"] and \
			  individual_1["f_cost"] <= individual_2["f_cost"] ) ):
			return True
		return False 

def non_dominated_sort():
	S = []
	N = []
	rank = []
	frontiers = [[]]
	
	for i in range( len( data ) ):
		S.append([])
		N.append( 0 )
		rank.append( 0 )

	for p in  range(len( data )) :
		for q in range(len( data )):
			if( dominate(data[p], data[q]) ):
				S[p].append(q)
			elif( dominate( data[q], data[p]) ):
				N[p] += 1
		if(N[p] == 0):
			rank[p] = 0
			frontiers[0].append(p)
	i = 0
	while( frontiers[i] != [] ):
		Q = []
		for p in frontiers[i]:
			for q in S[p]:
				N[q] -= 1
				if( N[q] == 0 ):
					rank[q] = i+1
					Q.append(q)
		i += 1
		frontiers.append( Q )
	print(frontiers)
	del frontiers[len(frontiers)-1]
	return frontiers

def non_dominated_sort_tsp():
	S = []
	N = []
	rank = []
	frontiers = [[]]
	
	for i in range( len( data ) ):
		S.append([])
		N.append( 0 )
		rank.append( 0 )

	for p in  range(len( data )) :
		for q in range(len( data )):
			if( dominate_tsp(data[p], data[q]) ):
				S[p].append(q)
			elif( dominate_tsp( data[q], data[p]) ):
				N[p] += 1
		if(N[p] == 0):
			rank[p] = 0
			frontiers[0].append(p)
	i = 0
	while( frontiers[i] != [] ):
		Q = []
		for p in frontiers[i]:
			for q in S[p]:
				N[q] -= 1
				if( N[q] == 0 ):
					rank[q] = i+1
					Q.append(q)
		i += 1
		frontiers.append( Q )
	del frontiers[len(frontiers)-1]
	return frontiers

def crowding_distance( frontiers ):
	distances = dict()
	for f in frontiers:
		distance = [ 0. ] * len(f)
		for m in range( n_functions ):
			m_sorted = [ [i, data[i]["fitness_"+str(m+1)]] for i in f ]
			m_sorted = sorted( m_sorted, key=lambda x: x[1] )
			if( len(m_sorted) > 1):
				distance[0] = np.inf
				distance[ len(f)-1 ] = np.inf

				if( len(m_sorted) > 2):
					m_max = max(m_sorted, key=lambda item: item[1])[1]
					m_min = min(m_sorted, key=lambda item: item[1])[1]
					if( m_max - m_min == 0):
						divisor = 10e-5
					else:
						divisor = m_max - m_min
					for k in range(1, len(f)-1 ):
						distance[k] += (m_sorted[k+1][1] - m_sorted[k-1][1]) / \
									   (divisor)
			else:
				distance[0] = 0
		for i in range( len(f) ):
			distances[f[i]] = distance[i]
	return distances

def crowding_distance_tsp( frontiers ):
	distances = dict()
	for f in frontiers:
		distance = [ 0. ] * len(f)
		for m in range( n_functions ):
			if( m == 0):
				m_sorted = [ [i, data[i]["f_distance"]] for i in f ]
			elif( m == 1): 
				m_sorted = [ [i, data[i]["f_cost"]] for i in f ]
			m_sorted = sorted( m_sorted, key=lambda x: x[1] )
			if( len(m_sorted) > 1):
				distance[0] = np.inf
				distance[ len(f)-1 ] = np.inf

				if( len(m_sorted) > 2):
					m_max = max(m_sorted, key=lambda item: item[1])[1]
					m_min = min(m_sorted, key=lambda item: item[1])[1]
					if( m_max - m_min == 0):
						divisor = 10e-5
					else:
						divisor = m_max - m_min
					for k in range(1, len(f)-1 ):
						distance[k] += (m_sorted[k+1][1] - m_sorted[k-1][1]) / \
									   (divisor)
			else:
				distance[0] = 0
		for i in range( len(f) ):
			distances[f[i]] = distance[i]
	return distances

def crowded_tournament_selection(frontiers, distances):
	global data
	new_data = []
	front = dict()
	c = 0
	for i in range( len( frontiers) ):
		for j in range( len(frontiers[i]) ):
			front[c] = frontiers[i][j]
			c+=1
	tmp = list( range( len(data) ) )
	while( len(new_data) < population_size ):
		perm_tmp = list(np.random.permutation( tmp ))
		# print( perm_tmp )
		rand_1 = perm_tmp[0]
		rand_2 = perm_tmp[1]
		# print( rand_1 )
		# print( rand_2 )
		if( front[rand_1] > front[rand_2] ):
			new_data.append( data[rand_1] )
			tmp.remove(rand_1)
		elif( front[rand_1] < front[rand_2] ):
			new_data.append( data[rand_2] )
			tmp.remove(rand_2) 
		elif( front[rand_1] == front[rand_2] ):
			if( distances[rand_1] >= distances[rand_2] ):
				new_data.append( data[rand_1] )
				tmp.remove(rand_1) 
			else:
				new_data.append( data[rand_2] )
				tmp.remove(rand_2) 

	return new_data

def crowded_selection_tsp( frontiers, distances):
	global data
	new_data = []
	
	while( True ):
		for i in frontiers:
			for j in i:
				new_data.append( data[j] )
				# print(len(new_data))
				if( len(new_data) >= population_size ):
					# print("asd")
					return new_data

def minimize_F():
	global data
	iteration = 0

	generate_population()
	evaluate_population()

	while( iteration <= n_iterations ):

		print("iteration #", iteration)

		for i in range(offspring_size):

			while(True):
				m_individual = BLX_crossover( tournament_selection(), \
											  tournament_selection() )
				if( random.random() <= mutation_prob ):
					uniform_mutation( m_individual )
				if( valid_individual( m_individual ) ):
					break	

			data.append( m_individual )
			data.append( m_individual )

		print("data\t", len(data))

		frontiers = non_dominated_sort()
		distances = crowding_distance( frontiers )

		# new_data = crowded_tournament_selection( frontiers, distances )
		new_data = crowded_selection_tsp( frontiers, distances )
		data = []
		data = copy.deepcopy( new_data )


		iteration += 1

		for i in data:
			print("fitness_1  ", i["fitness_1"], "\tfitness_2  ", i["fitness_2"])

	print(data)
	plt.plot([ i["fitness_1"] for i in data ], \
				[i["fitness_2"] for i in data], 'ro')
	# plt.axis([0, 6, 0, 20])
	plt.show()

def minimize_tsp():
	global data
	iteration = 0

	generate_population_tsp()
	evaluate_population_tsp()

	while( iteration <= n_iterations ):

		print("iteration #", iteration)

		for i in range( offspring_size ):

			# while(True):
			m_individual = PBX_crossover( tournament_selection_tsp(), \
										  tournament_selection_tsp() )
			if( random.random() <= mutation_prob ):
				mutation_tsp( m_individual[0] )
			if( random.random() <= mutation_prob ):
				mutation_tsp( m_individual[1] )
				# if( valid_individual( m_individual[0] ) and \
				# 		valid_individual( m_individual[1] ) ):
				# 	break	

			data.append( m_individual[0] )
			data.append( m_individual[1] )

		# print("data\t", len(data))

		frontiers = non_dominated_sort_tsp()
		distances = crowding_distance_tsp( frontiers )

		new_data = crowded_selection_tsp( frontiers, distances )
		data = []
		data = copy.deepcopy( new_data )

		print(frontiers)

		iteration += 1

		for i in data:
			print("distance  ", i["f_distance"], "\tcost  ", i["f_cost"])

	print(data)
	plt.plot([ i["f_distance"] for i in data ], \
				[i["f_cost"] for i in data], 'ro')
	# plt.axis([0, 6, 0, 20])
	plt.show()

if __name__ == "__main__":
	minimize_tsp()
	# minimize_F()
	# o_1 = [0.1, 0.4, 0.51, 0.58, 0.92, 0.12, 0.26, 0.37, 0.48, 0.61]
	# o_2 = [0.6, 0.65, 0.9, 0.21, 0.97, 0.49, 0.82, 0.3, 0.41, 0.61]

	# data = []
	# for i in range(10):
	# 	data.append({"fitness_1": o_1[i], "fitness_2": o_2[i]})

	# print(data)
	# f = non_dominated_sort()
	# print(f)
	# data = [ for i in range(10) ]

	# generate_population()
	# evaluate_population()
	# asd = non_dominated_sort()
	# print(asd)
	# qwe = crowding_distance( asd )
	# print("D\t", qwe)
	# print("\n\n",data)

	# print( crowded_tournament_selection( asd, qwe ) )