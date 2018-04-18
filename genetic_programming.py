import numpy as np
import random
import math
import operator
from operator import itemgetter

operators = {   "+": operator.add, 
				"-": operator.sub,
				"*": operator.mul,
				"%": operator.mod,
				"/": operator.truediv	}

inputs =  [0, 0.1,   0.2,  0.3,   0.4,  0.5,   0.6,  0.7,   0.8,  0.9]
outputs = [0, 0.005, 0.02, 0.045, 0.08, 0.125, 0.18, 0.245, 0.32, 0.405]

# terminals = [-54, 1, 2]
# functions = ["+", "-", "*", "/"]
constant_numbers = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
terminals = [-54, -55]
# terminals = [-54, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
functions = ["+", "-", "*", "%", "/"]

# sol = [4, 2, 0, 0, 2, 8, 7]

terminals_size = len(terminals) - 1
functions_size = len(functions) - 1

n_population = 125
n_iterations = 20

reproduction_prob = 0.1
mutation_prob = 0.1
crossover_prob = 0.8

individual_size = 7							#	(/(* x x)(/ 2 1))
function_pattern = [1, 1, 0, 0, 1, 0, 0]  	#	1 for function; 0 for terminal

k_adversaries = 6
crossover_point = 4

data = []

def get_random_terminal():
	tmp = random.randint(0, terminals_size)
	if( tmp == 0):
		return tmp
	else:
		rand_1 = random.randint(0, len(constant_numbers) - 1)
		return constant_numbers[rand_1]

def get_random_function():
	return random.randint(0, functions_size)
 
def function_fitness( x ):
	MSE = 0.
	for i in range(len(inputs)):
		c_1 = inputs[i] if x[2]==-54 else x[2]
		c_2 = inputs[i] if x[3]==-54 else x[3]
		c_3 = inputs[i] if x[5]==-54 else x[5]
		c_4 = inputs[i] if x[6]==-54 else x[6]

		if( (functions[x[1]] == "/" or functions[x[1]] == "%") and c_2 == 0):
			a = 0.
		else:
			a = operators[functions[x[1]]] ( c_1, c_2)
		# print("A ", a)
		if( (functions[x[4]] == "/" or functions[x[4]] == "%") and c_4 == 0):
			b = 0.
		else:
			b = operators[functions[x[4]]] ( c_3, c_4)
		# print("B ", b)
		if( (functions[x[0]] == "/" or functions[x[0]] == "%") and b == 0):
			c = 0
		else:
			c = operators[functions[x[0]]] (a, b)
		# print("C ", c)
		MSE += abs( outputs[i] - c )**2
	return  MSE / len(inputs)

def generate_population(n_population, individual_size):
	population = []
	for i in range(n_population):
		tmp_function = []
		for j in function_pattern:
			if( j == 1):
				tmp_function.append(get_random_function())
			else:
				tmp_function.append(get_random_terminal())
		population.append(tmp_function)
	print("Generating population:")
	print('\n'.join(' '.join(map(str,i)) for i in population))
	
	for i in population:
		data.append([i, [0] ])
	print_symbols()

def eval_population():
	fitness = []
	for i in data:
		i[1][0] = ( function_fitness( i[0] ) )

def print_symbols():
	for i in data:
		for j, k in zip(i[0], function_pattern):
			if( k == 1 ):
				print(functions[j], end='  ')
			else:
				if(j == 0):
					print('x', end='  ')
				else:
					print(j, end='  ')
		print("\t"+str( i[1]) )

def get_parent(k_adversaries):
	""" picks randomly k elements from the population taking the winner the one with
		more fitness value
	"""
	pool = len(data)
	selected = []

	for i in range(k_adversaries):
		tmp = random.randint(0, pool-1)
		selected.append([tmp, data[tmp][1]])

	index, value = min(enumerate([i[1] for i in selected]), key=itemgetter(1))
	parent_index = selected[index][0]
	return parent_index

def tournament_selection(k_adversaries):
	return get_parent(k_adversaries)

def crossover( parent_index_1, parent_index_2 ):
	offpsring = []

	crossover_point = random.randint(1, individual_size-2)

	mother_cromosome = data[parent_index_1][0]
	father_cromosome = data[parent_index_2][0]

	son_1 = np.concatenate([mother_cromosome[0:crossover_point], \
						father_cromosome[crossover_point:]])
	son_2 = np.concatenate([father_cromosome[0:crossover_point], \
						mother_cromosome[crossover_point:]])

	offpsring.append(list(son_1))
	offpsring.append(list(son_2))
	return offpsring

def mutation( parent_index ):
	random_index = random.randint(0, individual_size-1)
	tmp = list(data[parent_index][0])
	if( function_pattern[random_index] ):
		tmp[random_index] = get_random_function()
	else:
		tmp[random_index] = get_random_terminal()
	return tmp

def ms_ra(parent_index):
	son = list(data[parent_index][0])
	H = 12
	while(True):
		there_is = False
		son_fitm = function_fitness(son)
		for i in range(H):
			rand = random.randint(0,individual_size-1)
			son_tmp = son[:]
			if( function_pattern[rand] == 1):
				son_tmp[rand] = get_random_function()
			else:
				son_tmp[rand] = get_random_terminal()
			if( son_fitm  > function_fitness(son_tmp) ):
				son = son_tmp
				there_is = True
				break
		if(there_is == False):
			return son

def sorting_population():
	""" reduces the number of the population just surviving the strongest
	"""
	global data
	# print("Selecting next population")
	# print('\n'.join(' '.join(' '.join(map(str, j))for j in i)for i in data))
	tmp = []
	for i,x in zip(data,range(len(data))):
		tmp.append([x, i[1]])
	tmp = sorted(tmp, key=itemgetter(1), reverse=False)
	ms = []
	for i in range(n_population):
		ms.append(data[tmp[i][0]])
	data = ms

def genetic_programing():

	generate_population(n_population, individual_size)
	eval_population()

	global data


	L2 = (crossover_prob + reproduction_prob)
	tol = 1e-12
	counter = 0
	while(True):
		pass
		print("iteration: ", counter)
	# for i in range(n_iterations):
	# 	print("\n\nIteration " + str(i) + " :\n\n")
		# print('\n'.join('\t'.join(' '.join(map(str, j))for j in i)for i in data ))
		# print_symbols()

		tmp = []
		while(True):
			if( len(tmp) >= n_population ):
				break
			genetic_operation = random.uniform(0, 1)
			if(genetic_operation <= crossover_prob):
				#crossover
				parent_1 = tournament_selection(k_adversaries)
				parent_2 = tournament_selection(k_adversaries)
				offspring = crossover(parent_1, parent_2)
				for i in offspring:
					tmp.append([i, [function_fitness(i)]])
			elif( crossover_prob < genetic_operation <= L2):
				#reproduction
				parent_1 = tournament_selection(k_adversaries)
				# son = mutation(parent_1)
				# tmp.append([ son, [function_fitness(son)]])
				tmp.append(data[parent_1])
			elif( L2 < genetic_operation <= 1.0 ):
				#mutation
				parent_1 = tournament_selection(k_adversaries)
				# son = ms_ra(parent_1)
				son = mutation(parent_1)
				tmp.append([ son, [function_fitness(son)]])

		data = []
		for i in tmp:
			data.append(i)
		counter+=1
		sorting_population()
		if (data[0][1][0] < tol or counter == 1125):
			break

	print("RESULTS")
	print_symbols()

genetic_programing()

# print(function_fitness(sol))

# def ms_asdasd(n_population, individual_size):
# 	population = []
# 	c = 0
# 	while(True):
# 		tmp_function = []
# 		for j in function_pattern:
# 			if( j == 1):
# 				tmp_function.append(get_random_function())
# 			else:
# 				tmp_function.append(get_random_terminal())
# 		print(tmp_function, end='\t')
# 		print(sol)

# 		print(c)
# 		c += 1
# 		if( tmp_function == sol):
# 			print("GOT YOU")
# 			break

# ms_asdasd(n_population, individual_size)