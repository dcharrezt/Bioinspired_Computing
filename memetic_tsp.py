import numpy as np
from bitstring import BitArray
import random
from operator import itemgetter
import math


roads = [[0,12,3,23,1,5,23,56,12,11],[12,0,9,18,3,41,45,5,41,27],
			[3,9,0,89,56,21,12,48,14,29],[23,18,89,0,87,46,75,17,50,42],
			[1,3,56,87,0,55,22,86,14,33],[5,41,21,46,55,0,21,76,54,81],
			[23,45,12,75,22,21,0,11,57,48],[56,5,48,17,86,76,11,0,63,24],
			[12,41,14,50,14,54,57,63,0,9],[11,27,29,42,33,81,48,24,9,0]]


cities = [i for i in range(len(roads))]

n_population = 100
cromosome_size = len(roads)

iterations = 20
cross_prob = 0.9
mutation_prob = 0.05
k_adversaries = 3

H = 5
N = 3

pmx_point_1 = 1
pmx_point_2 = 3

data = []


def OBX_crossover(mother_index, father_index):
	offpsring = []

	mother_cromosome = data[mother_index][0]
	father_cromosome = data[father_index][0]

	son_1, son_2 = [], []
	mask = np.random.randint(2, size=cromosome_size)

	for i, j in zip(range(cromosome_size), mask):
		if(j == 1):
			son_1.append(mother_cromosome[i])
			son_2.append(father_cromosome[i])
		else:
			son_1.append(-1)
			son_2.append(-1)

	print(son_1)
	print(son_2)

	for i in mother_cromosome:
		if (i not in son_2):
			tmp = son_2.index(-1)
			son_2[tmp] = i

	for i in father_cromosome:
		if (i not in son_1):
			tmp = son_1.index(-1)
			son_1[tmp] = i


	print(*son_1, sep='')
	print(*son_2, sep='')

	offpsring.append(son_1)
	offpsring.append(son_2)
	return offpsring

def PBX_crossover(mother_index, father_index):
	offpsring = []

	mother_cromosome = data[mother_index][0]
	father_cromosome = data[father_index][0]

	son_1, son_2 = [], []

	mask = np.random.randint(2, size=cromosome_size)

	for i, j in zip(range(cromosome_size), mask):
	
		if(j == 1):
			son_1.append(father_cromosome[i])
			son_2.append(mother_cromosome[i])
		else:
			son_1.append(-1)
			son_2.append(-1)

	print(son_1)
	print(son_2)

	for i in mother_cromosome:
		if (i not in son_1):
			tmp = son_1.index(-1)
			son_1[tmp] = i

	for i in father_cromosome:
		if (i not in son_2):
			tmp = son_2.index(-1)
			son_2[tmp] = i


	print(*son_1, sep='')
	print(*son_2, sep='')

	offpsring.append(son_1)
	offpsring.append(son_2)
	return offpsring

def PMX_crossover(mother_index, father_index):
	offpsring = []

	mother_cromosome = data[mother_index][0]
	father_cromosome = data[father_index][0]

	son_1, son_2 = mother_cromosome, father_cromosome

	print(*son_1, sep='')
	print(*son_2, sep='')

	for i in range(pmx_point_1, pmx_point_2+1):
		tmp_1 = np.where(son_2 == son_1[i])
		tmp_2 = np.where(son_1 == son_2[i])

		tmp_1 = tmp_1[0]
		tmp_2 = tmp_2[0]

		son_1[i], son_2[i] = son_2[i], son_1[i]
		son_2[tmp_1], son_1[tmp_2] = son_1[i], son_2[i]


		print(*son_1, sep='')
		print(*son_2, sep='')

		print()


	offpsring.append(son_1)
	offpsring.append(son_2)
	return offpsring

def CX_crossover(mother_index, father_index):
	offpsring = []

	mother_cromosome = data[mother_index][0]
	father_cromosome = data[father_index][0]

	son_1 = [ -1 for i in range(cromosome_size)]
	son_2 = [ -1 for i in range(cromosome_size)]

	print(*son_1, sep='')
	print(*son_2, sep='')
	print()

	tmp = 0
	son_1[tmp] = mother_cromosome[tmp]

	while(True):
		tmp_index = np.where(mother_cromosome == father_cromosome[tmp])[0][0]
		son_1[tmp_index] = mother_cromosome[tmp_index]
		tmp = tmp_index
		if(father_cromosome[tmp] in son_1):
			break

	tmp = 0
	son_2[tmp] = mother_cromosome[tmp]

	while(True):
		tmp_index = np.where(mother_cromosome == father_cromosome[tmp])[0][0]
		son_2[tmp_index] = mother_cromosome[tmp_index]
		tmp = tmp_index
		if(father_cromosome[tmp] in son_2):
			break

	for i in range(cromosome_size):
		if( son_1[i] == -1):
			son_1[i] = mother_cromosome[i]
		if( son_2[i] == -1):
			son_2[i] = father_cromosome[i]

	print(*son_1, sep='')
	print(*son_2, sep='')

	print()

	offpsring.append(son_1)
	offpsring.append(son_2)
	return offpsring

def function_fitness(x):
	total_length = 0
	for i in range(len(x)-1):
		total_length += roads[x[i]][x[i+1]]
	return (-1) * total_length

def get_random_cromosome( cities ):
	return np.random.permutation( cities )

def generate_population(n_population, cromosome_size):
	""" Receives as inputs the individuals in a population and cromosome size
    	then generates the population that is saved in the global varible data
    """
	population =  []
	for i in range(0, n_population * 3):
		population.append(get_random_cromosome( cities ))
	print("Generating population:")
	print('\n'.join(' '.join(map(str,i)) for i in population))
	for i in population:
		data.append([i])

def eval_population():
	""" reads cromosomes from global variable data and saves to data the returns
		of the function fitness_fuction that uses cromosomes as input
    """

	fitness = []
	for i in data:
		i.append([function_fitness( i[0] )])
	

def get_parent(k_adversaries):
	""" picks randomly k elements from the population taking the winner the one with
		more fitness value
	"""
	pool = len(data)
	selected = []

	for i in range(k_adversaries):
		tmp = random.randint(0, pool-1)
		selected.append([tmp, data[tmp][1]])

	index, value = max(enumerate([i[1] for i in selected]), key=itemgetter(1))
	parent_index = selected[index][0]
	return parent_index

def tournament_selection(k_adversaries):
	""" check if the mother and parent is the same tries again to finally return
		the selected indexes
	"""
	selected = []

	print("Selecting by tournament:")
	while(True):
		mother_index = get_parent(k_adversaries)
		father_index = get_parent(k_adversaries)
		if( mother_index != father_index):
			selected.append(mother_index)
			selected.append(father_index)
			break
	print("Mother: " + str(mother_index))
	print("Father: " + str(father_index))

	return selected


def selecting_next_population():
	""" reduces the number of the population just surviving the strongest
	"""
	global data
	print("Selecting next population")
	print('\n'.join(' '.join(' '.join(map(str, j))for j in i)for i in data))
	tmp = []
	for i,x in zip(data,range(len(data))):
		tmp.append([x, i[1]])
	tmp = sorted(tmp, key=itemgetter(1), reverse=True)
	ms = []
	for i in range(n_population):
		ms.append(data[tmp[i][0]])
	data = ms

def first_option_mutation( son, H):
	best_son = son
	king = son
	while(True):

		fitness_king = fitness_fuction(king)
		better_than_son = False
		for i in range(H):
			rand_1 = random.randint(0,cromosome_size-1)
			rand_2 = random.randint(0,cromosome_size-1)
			print("H ==> "+str(i))
			print("Before mutation")
			print(son, end='')
			print(" Fitness: "+str(fitness_king))
			son[rand_1], son[rand_2] = son[rand_2], son[rand_1]
			print("After mutation:")
			new_tmp = function_fitness(son)
			print(son, end='')
			print(" Fitness: "+str(new_tmp))
			if( abs(new_tmp) < abs(tmp)):
				best_son = son
				better_than_son = True
				break
		if(better_than_son):
			tmp = function_fitness(son)
		else:
			return best_son

def ms_ra(son, H):
	while(True):
		there_is = False
		son_fitm = function_fitness(son)
		for i in range(H):
			rand_1 = random.randint(0,cromosome_size-1)
			rand_2 = random.randint(0,cromosome_size-1)
			son_tmp = son[:]
			son_tmp[rand_1], son_tmp[rand_2] = son[rand_2], son[rand_1]
			if( son_fitm  < function_fitness(son_tmp) ):
				son = son_tmp
				there_is = True
				break

		if(there_is == False):
			return son

def genetic_algorithm():
	""" handles the flow between functions and counts iterations
	"""
	global data
	generate_population(n_population, cromosome_size)
	eval_population()
	selecting_next_population()

	for i in range(iterations):
		print("\n\nIteration " + str(i) + " :\n\n")
		print("Evaluating individuals:")
		print('\n'.join(' '.join(' '.join(map(str, j))for j in i)for i in data ))
		tmp = []
		while(True):
			if( len(data) + len(tmp) >= n_population*2):
				break
			if( random.uniform(0,100) <= cross_prob*100 ): # Crossove Prob
				selected = tournament_selection(k_adversaries)
				offspring = PBX_crossover(selected[0], selected[1])
				for son in offspring:
					print("iterations "+str(i)+" son n#: "+str(len(tmp)))
					print(str(function_fitness(son)) + "--->", end='')
					son = ms_ra(son, H)
					print(str(function_fitness(son)))
					print("Son to db:")
					print(son)
					tmp.append([son, [function_fitness(son) ]])
		for i in tmp:
			data.append(i)
		selecting_next_population()
	print("Evaluating individuals:")
	print('\n'.join(' '.join(' '.join(map(str, j))for j in i)for i in data ))



genetic_algorithm()