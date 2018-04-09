import numpy as np
from bitstring import BitArray
import random
from operator import itemgetter
import math

cities = [0, 1, 2, 3, 4]
roads = dict()

roads[0,1] = 2
roads[0,2] = 2
roads[0,3] = 1
roads[0,4] = 4

roads[1,0] = 2
roads[1,2] = 3
roads[1,3] = 2
roads[1,4] = 3

roads[2,0] = 2
roads[2,1] = 3
roads[2,3] = 2
roads[2,4] = 2

roads[3,0] = 1
roads[3,1] = 2
roads[3,2] = 2
roads[3,4] = 4

roads[4,0] = 4
roads[4,1] = 3
roads[4,2] = 2
roads[4,3] = 4

# roads = [ [0,1,2], [0,2,2], [0,3,1], [0,4,4], \
# 		  [1,0,2], [1,2,3], [1,3,2], [1,4,3], \
# 		  [2,0,2], [2,1,3], [2,3,2], [2,4,2], \
# 		  [3,0,1], [3,1,2], [3,2,2], [3,4,4], \
# 		  [4,0,4], [4,1,3], [4,2,2], [4,3,4] ]

n_population = 5
cromosome_size = len(cities)

iterations = 3
cross_prob = 0.9
mutation_prob = 0.05
k_adversaries = 3

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

	son_1, son_2 = [], []

	print(*son_1, sep='')
	print(*son_2, sep='')
	print()

	for i in range(len(cromosome_size)):
		son_1.







def function_fitness(x):
	total_length = 0
	for i in range(len(x)-1):
		total_length += roads[x[i], x[i+1]]
	return (-1) * total_length

def get_random_cromosome( cities ):
	return np.random.permutation( cities )

def generate_population(n_population, cromosome_size):
	""" Receives as inputs the individuals in a population and cromosome size
    	then generates the population that is saved in the global varible data
    """
	population =  []
	for i in range(0, n_population):
		population.append(get_random_cromosome( cities ))
	print("Generating population:")
	print('\n'.join('  '.join(map(str,i)) for i in population))
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
	print('\n'.join(' '.join('                  '.join(map(str, j))for j in i)for i in data))
	tmp = []
	for i,x in zip(data,range(len(data))):
		tmp.append([x, i[1]])
	tmp = sorted(tmp, key=itemgetter(1), reverse=True)
	ms = []
	for i in range(n_population):
		ms.append(data[tmp[i][0]])
	data = ms

def genetic_algorithm():
	""" handles the flow between functions and counts iterations
	"""
	global data
	generate_population(n_population, cromosome_size)
	eval_population()

	for i in range(iterations):
		print("\n\nIteration " + str(i) + " :\n\n")
		print("Evaluating individuals:")
		print('\n'.join(' '.join('   '.join(map(str, j))for j in i)for i in data ))
		tmp = []
		while(True):
			if( len(data) + len(tmp) >= n_population*2):
				break
			if( random.uniform(0,100) <= cross_prob*100 ): # Crossove Prob
				selected = tournament_selection(k_adversaries)
				offspring = PBX_crossover(selected[0], selected[1])
				for son in offspring:
					if( random.uniform(0,100) <= mutation_prob*100 ): # Mutation Prob
					 	print("Mutation")
					 	rand_1 = random.randint(0,cromosome_size-1)
					 	rand_2 = random.randint(0,cromosome_size-1)
					 	print(son)
					 	son[rand_1], son[rand_2] = son[rand_2], son[son_1]
					 	print(son)
					#   mutation for 0 and 1
					# 	tmp_son = son
					# 	son[random.randint(0,len(son)-1)] = 1
					# 	print(tmp_son)
					# 	print(son)
					tmp.append([son, [function_fitness(son) ]])

		for i in tmp:
			data.append(i)
		selecting_next_population()
	print("Evaluating individuals:")
	print('\n'.join(' '.join('              '.join(map(str, j))for j in i)for i in data ))



genetic_algorithm()