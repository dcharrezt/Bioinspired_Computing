import numpy as np
import random
import copy
import matplotlib.pyplot as plt

n_objectives = 2
n_iterations = 300
population_size = 50
offspring_size = 10

crossover_prob = 0.75
mutation_prob = 0.1

n_adversaries = 2 	# for tournament selection

path_dataset_cost = "datasets/small_cost.txt"
path_dataset_delay = "datasets/small_delay.txt"

data = []
cost_matrix = []
delay_matrix = []

solution_len = 0
max_cab_capacity = 4
n_cabs = 0
n_passengers = 0

def read_data():
	global n_cabs
	global n_passengers
	global solution_len

	with open(path_dataset_cost, 'r') as f:
		for line in f:
			cost_matrix.append( list( [float(n) for n in line.split()] ) )

	with open(path_dataset_delay, 'r') as f:
		for line in f:
			delay_matrix.append( list( [float(n) for n in line.split()] ) )

	n_cabs = len( cost_matrix )
	n_passengers = len( cost_matrix )
	solution_len = n_cabs + n_passengers

def generate_solution():
	sol = list( range(0, n_passengers) )
	cabs = [-1] * n_cabs
	sol += cabs
	random.shuffle( sol )
	return { "solution": sol, "cost": np.inf, "delay": np.inf }

def generate_population():
	for i in range( population_size ):
		data.append( generate_solution() )

def fix_solutions( batch ):
	for sol in batch:
		something_fishy = True
		while( something_fishy ):
			something_fishy = False
			passenger_counter = 0
			for i in range( solution_len ):
				if sol["solution"][i] == -1 or i == solution_len-1:
					if passenger_counter <= max_cab_capacity:
						passenger_counter = 0
					else:
						something_fishy = True
						rand = random.randint(2,passenger_counter-1)
						empty_cab_index = -1

						for j in range( solution_len-1 ):
							if( sol["solution"][j]==-1 and 
											sol["solution"][j+1]==-1 ):
								empty_cab_index = j+1

						first_passenger_index = i-passenger_counter
						# print("First, ", sol["solution"])
						if( empty_cab_index < first_passenger_index ):
							first_passenger_index -= 1
						del sol["solution"][empty_cab_index]
						sol["solution"].insert( first_passenger_index + rand, -1 )
						passenger_counter = 0
						i = solution_len
				else:
					passenger_counter += 1

	for i in data:
		if i["solution"].count(-1) > 10:
			print("BUG DETECTED----------------")
			print(i)

def evaluate_population():
	for i in data:
		i["cost"] = fitness_cost( i )
		i["delay"] = fitness_delay( i )

def fitness_cost( solu ):
	total_cost = 0
	for i in range( solution_len -1 ):
		if solu["solution"][i] != -1 and solu["solution"][i+1] != -1:
			total_cost += cost_matrix[ solu["solution"][i] ] \
											[ solu["solution"][i+1] ]
	return total_cost

def fitness_delay( solu ):
	total_delay = 0
	for i in range( solution_len -1 ):
		if solu["solution"][i] != -1 and solu["solution"][i+1] != -1:
			total_delay += delay_matrix[ solu["solution"][i] ] \
											[ solu["solution"][i+1] ]
	return total_delay

def PBX_crossover( solution_1, solution_2 ):
	offspring = []
	parent_1 = solution_1["solution"]
	parent_2 = solution_2["solution"]
	son_1, son_2 = [], []
	tmp_1, tmp_2 = [], []
	mask = np.random.randint(2, size=len(parent_1))
	for i, j in zip(range(len(parent_1)), mask):
		if(j == 1):
			son_1.append(parent_2[i])
			son_2.append(parent_1[i])
		else:
			son_1.append(-2)
			son_2.append(-2)

	for i in parent_1:
		if (i not in son_1) or (son_1.count(-1) < n_passengers and i==-1):
			son_1[son_1.index(-2)] = i

	for i in parent_2:
		if (i not in son_2) or (son_2.count(-1) < n_passengers and i==-1):
			son_2[son_2.index(-2)] = i

	new_individual_1 = { "solution": son_1 }
	new_individual_2 = { "solution": son_2 }
	fix_solutions( [new_individual_1, new_individual_2] )
	new_individual_1["cost"] = fitness_cost( new_individual_1 )
	new_individual_2["delay"] = fitness_delay( new_individual_2 )
	new_individual_1["cost"] = fitness_cost( new_individual_1 )
	new_individual_2["delay"] = fitness_delay( new_individual_2 )
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
	sums = [ i["cost"]+i["delay"] for i in tmp ]
	m_min = min( sums )
	return data[adversaries[sums.index(m_min)]] 

def tournament_selection():
	adversaries = np.random.permutation( list( range( population_size ) ) )
	tmp = [ data[i] for i in adversaries[:n_adversaries]]
	sums = [ i["fitness_1"]+i["fitness_2"] for i in tmp ]
	m_min = min( sums )
	return data[adversaries[sums.index(m_min)]] 

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
		for m in range( n_objectives ):
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
		for m in range( n_objectives ):
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
	plt.show()

def NSGAII_algorithm():
	global data
	iteration = 0
	read_data()
	generate_population()


	for i in data:
		if i["solution"].count(-1) > 10:
			print("BUG DETECTED ++++++++++++++++++++++++++")
			print(i)

	fix_solutions( data )


	evaluate_population()
	for i in range( n_iterations ):
		print("+++ Iteration ", i)
		for i in range( offspring_size ):
			offspring = PBX_crossover()

if __name__ == "__main__":
	NSGAII_algorithm()

	# a = PBX_crossover( data[0], data[1] )
	# print( a )

