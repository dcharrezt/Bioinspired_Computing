import numpy as np
from bitstring import BitArray
import random
from operator import itemgetter

n_population = 5
cromosome_size = 10
iterations = 30
cross_prob = 0.9
mutation_prob = 0.05
k_adversaries = 3

mask = np.random.randint(2, size=cromosome_size)
print("Mask: ")
print(*mask, sep='')


data = []

def fitness_function(x, y):
	"""returns function of x"""
	return x-y

def generate_population(n_population, cromosome_size):
	""" Receives as inputs the individuals in a population and cromosome size
    	then generates the population that is saved in the global varible data
    """
	population =  []
	for i in range(0, n_population):
		population.append(np.random.randint(2, size=cromosome_size))
	print("Generating population:")
	print('\n'.join(''.join(map(str,i)) for i in population))
	for i in population:
		data.append([i])

def eval_population():
	""" reads cromosomes from global variable data and saves to data the returns
		of the function fitness_fuction that uses cromosomes as input
    """
	fitness = []
	for i in data:
		x = BitArray(i[0][:5])
		y = BitArray(i[0][5:])
		i.append([fitness_function(x.uint, y.uint)]) # x^2
	

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


def uniform_crossover(mask, mother_index, father_index):
	""" given a mask that is a string of 0's and 1's creates two new
		cromosomes, takes one cromosome of the mother if the mask element is
		1 and takes one cromosome of the father if the mask is 0, for the second
		son does it the other way around
	"""
	offpsring = []
	mother_cromosome = data[mother_index][0]
	father_cromosome = data[father_index][0]

	son_1, son_2 = [], []

	for i,x in zip(mask,range(len(mask))):
		if(i == 0):
			son_1.append(mother_cromosome[x])
			son_2.append(father_cromosome[x])
		if(i == 1):
			son_1.append(father_cromosome[x])
			son_2.append(mother_cromosome[x])

	print(*son_1, sep='')
	print(*son_2, sep='')

	offpsring.append(son_1)
	offpsring.append(son_2)
	return offpsring

def selecting_next_population():
	""" reduces the number of the population just surviving the strongest
	"""
	global data
	print("Selecting next population")
	print('\n'.join(' '.join(''.join(map(str, j))for j in i)for i in data))
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
		print('\n'.join(' '.join(''.join(map(str, j))for j in i)for i in data ))
		tmp = []
		while(True):
			if( len(data) + len(tmp) >= n_population*2):
				break
			if( random.uniform(0,100) <= cross_prob*100 ): # Crossove Prob
				selected = tournament_selection(k_adversaries)
				offspring = uniform_crossover(mask, selected[0], selected[1])
				for son in offspring:
					if( random.uniform(0,100) <= mutation_prob*100 ): # Mutation Prob
						print("Mutation")
						tmp_son = son
						son[random.randint(0,len(son)-1)] = 1
						print(tmp_son)
						print(son)
					x = BitArray(son[:5])
					y = BitArray(son[5:])
					tmp.append([son,[fitness_function(x.uint, y.uint)]])

		for i in tmp:
			data.append(i)
		selecting_next_population()
	print("Evaluating individuals:")
	print('\n'.join(' '.join(''.join(map(str, j))for j in i)for i in data ))



genetic_algorithm()