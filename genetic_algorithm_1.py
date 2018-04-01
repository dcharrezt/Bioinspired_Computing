import numpy as np
from bitstring import BitArray
import random
from operator import itemgetter


n_population = 4
cromosome_size = 5
iterations = 30
cross_prob = 0.9
cross_point = 3
mutation_prob = 0.05
# f(x) = x^2

data = []


def fitness_function(x):
	"""returns function of x"""
	return x*x

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
		b = BitArray(i[0])
		i.append([fitness_function(b.uint)]) # x^2
	

def get_parent(roulette_range):
	""" this function is used by roulette_selection to return to simulate the
		behavior of a roulette
	"""
	selected = random.uniform(0,100)
	for i in roulette_range:
		if( i[0] <= selected and selected <=i[1] ):
			index = roulette_range.index(i)
			break
	return index

def roulette_selection():
	""" using fitness values get the percents which are needed for the roulette
		algorithm, calls get_parent to get the selected father and mother
	"""
	selected = []

	total = sum([i[1][0] for i in data])
	percents = [i[1][0]*100/total for i in data]

	print("Selecting by Roulette:")
	print('\n'.join(' '.join(''.join( map(str, j)) for j in i) \
						 + ' - '+str(ms)  for i, ms in zip(data, percents) ))
	roulette_range = []
	tmp = 0
	for i in percents:
		roulette_range.append([tmp, i+tmp])
		tmp += i
	while(True):
		mother_index = get_parent(roulette_range)
		father_index = get_parent(roulette_range)
		if( mother_index != father_index):
			selected.append(mother_index)
			selected.append(father_index)
			break
	print("Mother: " + str(mother_index))
	print("Father: " + str(father_index))

	return selected

def one_point_crossover(cross_point, mother_index, father_index):
	""" given a point mixes mother cromosome and father cromosome
	"""
	offpsring = []

	print("One point crossover in "+str(cross_point))
	mother_cromosome = data[mother_index][0]
	father_cromosome = data[father_index][0]

	son_1 = np.concatenate([mother_cromosome[0:cross_point], \
						father_cromosome[cross_point:]])
	son_2 = np.concatenate([father_cromosome[0:cross_point], \
						mother_cromosome[cross_point:]])

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
				selected = roulette_selection()
				offspring = one_point_crossover(cross_point, selected[0], selected[1])
				for son in offspring:
					if( random.uniform(0,100) <= mutation_prob*100 ): # Mutation Prob
						print("Mutation")
						tmp_son = son
						son[random.randint(0,len(son)-1)] = 1
						print(tmp_son)
						print(son)
					b = BitArray(son)
					tmp.append([son,[fitness_function(b.uint)]])

		for i in tmp:
			data.append(i)
		selecting_next_population()
	print("Evaluating individuals:")
	print('\n'.join(' '.join(''.join(map(str, j))for j in i)for i in data ))



genetic_algorithm()