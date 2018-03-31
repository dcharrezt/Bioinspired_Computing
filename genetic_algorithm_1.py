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

data = []

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
		i.append([b.uint])
	

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

def one_point_crossover(cross_point, mother_index, father_index):
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
	print("Selecting next population")
	print('\n'.join(' '.join(''.join(map(str, j))for j in i)for i in data))
	tmp = []
	c = 0
	for i,x in zip(data,range(len(data))):
		tmp.append([x, i[1]])
	print("not sorted")
	print(tmp)
	tmp = sorted(tmp, key=itemgetter(1))
	print("sorted")
	print(tmp)
	ms = []
	for i in range(n_population):
		print(tmp[i][0])
		del data[tmp[i][0]]

def genetic_algorithm():

	generate_population(n_population, cromosome_size)
	eval_population()

	for i in range(iterations):
		print("Iteration " + str(i) + " :")
		print("Evaluating individuals:")
		print('\n'.join(' '.join(''.join(map(str, j))for j in i)for i in data ))

		while(True):

			if( len(data) >= n_population*2):
				break
			selected = roulette_selection()
			offspring = one_point_crossover(cross_point, selected[0], selected[1])
			for i in offspring:
				b = BitArray(i)
				data.append([i,[b.uint]])
		selecting_next_population()
	print("Evaluating individuals:")
	print('\n'.join(' '.join(''.join(map(str, j))for j in i)for i in data ))

			
	# selecting_next_population()

genetic_algorithm()