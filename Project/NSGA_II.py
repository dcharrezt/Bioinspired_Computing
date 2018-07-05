import numpy as np
import random
import copy
import matplotlib.pyplot as plt

n_objectives = 2
n_iterations = 200
population_size = 100
offspring_size = 50

crossover_prob = 0.75
mutation_prob = 0.1

n_adversaries = 8 	# for tournament selection

path_dataset_cost = "datasets/big_cost.txt"
path_dataset_delay = "datasets/big_delay.txt"

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

	n_cabs = len( cost_matrix ) -1
	n_passengers = len( cost_matrix ) -1
	solution_len = n_cabs + n_passengers

def generate_solution():
	sol = list( range(1, n_passengers+1) )
	cabs = [-1] * n_cabs
	sol += cabs
	random.shuffle( sol )
	return { "solution": sol, "cost": np.inf, "delay": np.inf }

def generate_avid_solution( avid_solution ):
	data.append({"solution":avid_solution,"cost":np.inf,"delay":np.inf})
	for i in range( 1, population_size ):
		tmp = list(avid_solution)
		n_modifications = random.randint( 1, solution_len/4 )
		for j in range( n_modifications ):
			rand_1 = random.randint( 1 , solution_len )
			rand_2 = random.randint( 1 , solution_len )
			while( rand_1 == rand_2 ):
				rand_2 = random.randint( 1, solution_len )
		tmp[rand_1], tmp[rand_2] = tmp[rand_2], tmp[rand_1]
		data.append({"solution":avid_solution,"cost":np.inf,"delay":np.inf})

def generate_population():
	for i in range( population_size ):
		data.append( copy.deepcopy(generate_solution()) )

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

def evaluate_population( batch ):
	for i in batch:
		pc = get_passengers_by_cab( i )
		i["cost"] = fitness_cost( pc )
		i["delay"] = fitness_delay( pc )

def get_passengers_by_cab( solution ):
	pass_by_cab = []
	tmp = []
	for i in range( len(solution["solution"]) ):
		if solution["solution"][i] != -1:
			tmp.append( solution["solution"][i] )
		if solution["solution"][i] == -1:
			pass_by_cab.append( tmp )
			tmp = []
	pass_by_cab.append(tmp)
	pass_by_cab = [x for x in pass_by_cab if x != []]
	return pass_by_cab
 
def fitness_cost( pc ):
	total_cost = 0
	for i in pc:
		if len(i) == 1:
			total_cost+= cost_matrix[0][i[0]]
		else:
			for j in range(len(i)-1):
				if j == 0:
					total_cost += cost_matrix[0][i[j]]
				total_cost += cost_matrix[i[j]][i[j+1]]
	return total_cost

def fitness_delay( pc ):
	total_delay = 0
	for i in pc:
		if len(i) == 1:
			total_delay+= delay_matrix[0][i[0]]
		else:
			for j in range(len(i)-1):
				if j == 0:
					total_delay += delay_matrix[0][i[j]]
				total_delay += delay_matrix[i[j]][i[j+1]]
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
	pc_1 = get_passengers_by_cab( new_individual_1 )
	pc_2 = get_passengers_by_cab( new_individual_2 )
	new_individual_1["cost"] = fitness_cost( pc_1 )
	new_individual_1["delay"] = fitness_delay( pc_1 )
	new_individual_2["cost"] = fitness_cost( pc_2 )
	new_individual_2["delay"] = fitness_delay( pc_2 )
	offspring.append( copy.deepcopy(new_individual_1) )
	offspring.append( copy.deepcopy(new_individual_2) )

	return offspring

def mutation( solution ):
	perm_tmp = list(np.random.permutation( n_passengers*2 ))
	solution["solution"][perm_tmp[1]], solution["solution"][perm_tmp[0]] = \
	 	solution["solution"][perm_tmp[0]], solution["solution"][perm_tmp[1]]

def tournament_selection():
	adversaries = np.random.permutation( list( range( population_size ) ) )
	tmp = [ data[i] for i in adversaries[:n_adversaries]]
	sums = [ i["cost"]+i["delay"] for i in tmp ]
	m_min = min( sums )
	return copy.deepcopy(data[adversaries[sums.index(m_min)]]) 

def dominance( solution_1, solution_2 ):
	if( ( solution_1["cost"] < solution_2["cost"] and \
		  solution_1["delay"] < solution_2["delay"] ) or \
		( solution_1["cost"] <= solution_2["cost"] and \
		  solution_1["delay"] < solution_2["delay"] ) or \
		( solution_1["cost"] < solution_2["cost"] and \
		  solution_1["delay"] <= solution_2["delay"] ) ):
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
			if( dominance(data[p], data[q]) ):
				S[p].append(q)
			elif( dominance( data[q], data[p]) ):
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
			if( m == 0):
				m_sorted = [ [i, data[i]["cost"]] for i in f ]
			elif( m == 1): 
				m_sorted = [ [i, data[i]["delay"]] for i in f ]
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
		if( front[rand_1] < front[rand_2] ):
			new_data.append( data[rand_1] )
			tmp.remove(rand_1)
		elif( front[rand_1] > front[rand_2] ):
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

def crowded_selection_tsp( frontiers):
	global data
	new_data = []
	
	while( True ):
		for i in frontiers:
			for j in i:
				new_data.append( copy.deepcopy(data[j]) )
				# print(len(new_data))
				if( len(new_data) >= population_size ):
					# print("asd")
					return new_data

def NSGAII_algorithm():
	global data
	iteration = 0
	read_data()
	generate_population()
	fix_solutions( data )
	evaluate_population( data )

	for i in range( n_iterations ):
		print("+++ Iteration ", i)
		for i in range( offspring_size ):
			offspring = PBX_crossover(tournament_selection(),tournament_selection())
			if( random.random() <= mutation_prob ):
				mutation( offspring[0] )
			if( random.random() <= mutation_prob ):
				mutation( offspring[1] )
			fix_solutions( offspring )
			evaluate_population( offspring )
			data.append( copy.deepcopy(offspring[0]) )
			data.append( copy.deepcopy(offspring[1]) )
		print(len(data))
		frontiers = non_dominated_sort()
		# distances = crowding_distance( frontiers )
		# new_data = crowded_tournament_selection( frontiers, distances )
		new_data = []
		new_data = crowded_selection_tsp( frontiers )
		# for i in frontiers[0]:
		# 	new_data.append( data[i] )
		data = []
		data = copy.deepcopy( new_data )
		print(frontiers)
		for i in data:
			print("cost  ", i["cost"], "\t delay  ", i["delay"])
	print(data)


if __name__ == "__main__":
	NSGAII_algorithm()
	frontiers = non_dominated_sort()

	pareto = []
	for i in frontiers[0]:
		pareto.append( data[i] )
	plt.plot([ i["cost"] for i in pareto ], \
				[i["delay"] for i in pareto], 'ro')
	plt.show()


