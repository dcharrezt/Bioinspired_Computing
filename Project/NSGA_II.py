import numpy as np
import random
import copy
import matplotlib.pyplot as plt

n_objectives = 2
n_iterations = 20
population_size = 5
offspring_size = 3

crossover_prob = 0.75
mutation_prob = 0.1

n_adversaries = 3 	# for tournament selection

# For small dataset
avid_cost = [1, 3, 2, 7, 6, 8, 4, 5, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0]
avid_delay = [6, 3, 4, 5, 7, 0, 9, 8, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]

path_dataset_cost = "datasets/small_cost.txt"
path_dataset_delay = "datasets/small_delay.txt"

# For large dataset
# avid_cost = [9,7,3,2,5,34,35,18,31,0,14,37,36,20,21,19,1,4,38,0,25,26,40,10, \
# 			39,6,8,13,16,0,30,12,32,33,17,44,29,11,27,0,28,22,41,15,42,43,23,24, \
# 			0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
# 			0,0,0,0,0,0,0]
# avid_delay = [1,39,0,2,0,5,3,4,0,7,32,0,8,0,9,0,10,0,11,0,12,0,13,0,14,0,15,0, \
# 			16,0,17,0,18,26,0,19,0,20,37,0,21,0,22,0,23,0,24,25,0,27,0,28,29,0, \
# 			30,0,31,34,0,33,0,35,6,0,36,0,38,0,40,0,41,42,43,0,44,0,0,0,0,0,0,0, \
# 			0,0,0,0,0,0,0,0]

# path_dataset_cost = "datasets/large_cost.txt"
# path_dataset_delay = "datasets/large_delay.txt"

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
	print("SOL ", solution_len)

def generate_solution():
	sol = list( range(1, n_passengers+1) )
	cabs = [-1] * n_cabs
	sol += cabs
	random.shuffle( sol )
	return { "solution": sol, "cost": np.inf, "delay": np.inf }

def generate_avid_solution():
	global avid_cost, avid_delay
	data.append({"solution":avid_cost,"cost":np.inf,"delay":np.inf})
	data.append({"solution":avid_delay,"cost":np.inf,"delay":np.inf})
	for i in range( 2, population_size ):
		if i < population_size/2:
			tmp = list(avid_cost)
			n_modifications = random.randint( 1, int(solution_len/4) )
			for j in range( n_modifications ):
				rand_1 = random.randint( 1 , solution_len-1 )
				rand_2 = random.randint( 1 , solution_len-1 )
				while( rand_1 == rand_2 ):
					rand_2 = random.randint( 1, solution_len-1 )
				tmp[rand_1], tmp[rand_2] = tmp[rand_2], tmp[rand_1]
			data.append({"solution":tmp,"cost":np.inf,"delay":np.inf})
		else:
			tmp = list(avid_delay)
			n_modifications = random.randint( 1, int(solution_len/4) )
			for j in range( n_modifications ):
				rand_1 = random.randint( 1 , solution_len-1 )
				rand_2 = random.randint( 1 , solution_len-1 )
				while( rand_1 == rand_2 ):
					rand_2 = random.randint( 1, solution_len-1 )
				tmp[rand_1], tmp[rand_2] = tmp[rand_2], tmp[rand_1]
			data.append({"solution":tmp,"cost":np.inf,"delay":np.inf})

def find_consecutives_zeros( sol ):
	for i in range(solution_len-1):
		if sol["solution"][i] == 0 and sol["solution"][i+1] == 0:
			return i

def corrective_function( batch):
	for sol in batch:
		while(True):
			seq_size = 0
			position = 0
			while((seq_size<=max_cab_capacity) and (position<solution_len)):
				if sol["solution"][position]!=0:
					seq_size+=1
				else:
					seq_size = 0
				position+=1
			if seq_size>max_cab_capacity:
				rand = random.randint(position-max_cab_capacity+1,position-1)
				index_zero = find_consecutives_zeros( sol )
				sol["solution"][index_zero], sol["solution"][rand] = \
					sol["solution"][rand], sol["solution"][index_zero]
			else:
				break

def crossover_PBX( solution_1, solution_2 ):
	parents = [ solution_1["solution"][:], solution_2["solution"][:]]
	offspring= [[],[]]
	for i in range(2):
		offspring[i] = [-1]*solution_len
		mask = np.random.randint(2, size=solution_len)
		for j in range( solution_len ):
			if mask[j] == 1:
				offspring[i][j] = parents[i][j]
		if i == 0:
			cp_parent = parents[1][:]
			for j in range(solution_len):
				if mask[j] == 1:
					cp_parent[cp_parent.index(parents[0][j])] = -1
			for j in range(solution_len):
				if cp_parent[j] != -1:
					offspring[i][offspring[0].index(-1)] = cp_parent[j]
		else:
			cp_parent = parents[0][:]
			for j in range(solution_len):
				if mask[j] == 1:
					cp_parent[cp_parent.index(parents[1][j])] = -1
			for j in range(solution_len):
				if cp_parent[j] != -1:
					offspring[i][offspring[1].index(-1)] = cp_parent[j]
	return [{"solution":offspring[0], "cost":np.inf, "delay":np.inf}, \
					{"solution":offspring[1], "cost":np.inf, "delay":np.inf}]

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
	rand_1 = random.randint(0, solution_len-1)
	rand_2 = random.randint(0, solution_len-1)
	while rand_1 == rand_2:
		rand_2 = random.randint(0, solution_len-1)
	solution["solution"][rand_1], solution["solution"][rand_2] = \
	 	solution["solution"][rand_2], solution["solution"][rand_1]

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
			new_data.append( copy.deepcopy(data[rand_1]) )
			tmp.remove(rand_1)
		elif( front[rand_1] > front[rand_2] ):
			new_data.append( copy.deepcopy(data[rand_2]) )
			tmp.remove(rand_2) 
		elif( front[rand_1] == front[rand_2] ):
			if( distances[rand_1] >= distances[rand_2] ):
				new_data.append( copy.deepcopy(data[rand_1]) )
				tmp.remove(rand_1) 
			else:
				new_data.append( copy.deepcopy(data[rand_2]) )
				tmp.remove(rand_2) 

	return new_data

def crowded_selection_tsp( frontiers):
	global data
	new_data = []
	while( True ):
		for i in frontiers:
			for j in i:
				new_data.append( copy.deepcopy(data[j]) )
				if( len(new_data) >= population_size ):
					return new_data

def NSGAII_algorithm():
	global data
	iteration = 0
	read_data()
	generate_avid_solution()
	corrective_function( data )
	evaluate_population( data )

	for i in range( n_iterations ):
		print("+++ Iteration ", i)
		for i in range( offspring_size ):
			offspring = crossover_PBX(tournament_selection(),tournament_selection())
			if( random.random() <= mutation_prob ):
				mutation( offspring[0] )
			if( random.random() <= mutation_prob ):
				mutation( offspring[1] )
			corrective_function( offspring )
			evaluate_population( offspring )
			data.append( copy.deepcopy(offspring[0]) )
			data.append( copy.deepcopy(offspring[1]) )
		frontiers = non_dominated_sort()
		distances = crowding_distance( frontiers )
		new_data = crowded_tournament_selection( frontiers, distances )
		# new_data = []
		# new_data = crowded_selection_tsp( frontiers )
		# for i in frontiers[0]:
		# 	new_data.append( data[i] )
		data = []
		data = copy.deepcopy( new_data )
		for i in data:
			print("cost  ", i["cost"], "\t delay  ", i["delay"])
	# print(data)


if __name__ == "__main__":
	NSGAII_algorithm()
	frontiers = non_dominated_sort()

	# pareto = []
	# for i in frontiers[0]:
	# 	pareto.append( data[i] )
	# plt.plot([ i["cost"] for i in pareto ], \
	# 			[i["delay"] for i in pareto], 'ro')
	# plt.show()


