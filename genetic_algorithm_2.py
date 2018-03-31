import numpy as np
from bitstring import BitArray
import random
from operator import itemgetter

n_population = 4
cromosome_size = 5
iterations = 30
cross_prob = 0.9
cross_point_1 = 1
cross_point_2 = 3
mutation_prob = 0.05

data = []

def fitness_function(x):
	return (-(x*x)/10) + 3*x

def generate_population(n_population, cromosome_size):
	population =  []
	for i in range(0, n_population):
		population.append(np.random.randint(2, size=cromosome_size))
	print("Generating population:")
	print('\n'.join(''.join(map(str,i)) for i in population))
	for i in population:
		data.append([i])

def eval_population():
	fitness = []
	for i in data:
		b = BitArray(i[0])
		i.append([fitness_function(b.uint)]) # x^2
	

def get_parent(roulette_range):
	selected = random.uniform(0,100)
	for i in roulette_range:
		if( i[0] <= selected and selected <=i[1] ):
			index = roulette_range.index(i)
			break
	return index

def roulette_selection():
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

def two_point_crossover(cross_point_1, cross_point_2, mother_index, father_index):
	offpsring = []

	print("Two point crossover between "+str(cross_point_1)+" and "+str(cross_point_2))
	mother_cromosome = data[mother_index][0]
	father_cromosome = data[father_index][0]

	son_1 = np.array(mother_cromosome[:cross_point_1].tolist() + \
					father_cromosome[cross_point_1:cross_point_2].tolist() +\
					mother_cromosome[cross_point_2:].tolist())

	son_2 = np.array(father_cromosome[:cross_point_1].tolist() + \
				mother_cromosome[cross_point_1:cross_point_2].tolist() +\
				father_cromosome[cross_point_2:].tolist())
	
	print(*son_1, sep='')
	print(*son_2, sep='')

	offpsring.append(son_1)
	offpsring.append(son_2)
	return offpsring

def selecting_next_population():
	global data
	print("Selecting next population")
	print('\n'.join(' '.join(''.join(map(str, j))for j in i)for i in data))
	tmp = []
	c = 0
	for i,x in zip(data,range(len(data))):
		tmp.append([x, i[1]])
	tmp = sorted(tmp, key=itemgetter(1), reverse=True)
	ms = []
	for i in range(n_population):
		ms.append(data[tmp[i][0]])
	data = ms

def genetic_algorithm():
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
				offspring = two_point_crossover(cross_point_1, cross_point_2, \
												selected[0], selected[1])
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