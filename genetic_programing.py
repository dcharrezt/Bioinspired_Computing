import numpy as np
import random
import math
import operator

operators = {   "+": operator.add, 
				"-": operator.sub,
				"*": operator.mul,
				"%": operator.mod,
				"/": operator.truediv}

inputs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
outputs = [120, 0.005, 0.02, 0.045, 0.08, 0.125, 0.18, 0.245, 0.32, 0.405]

terminals = [-54, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
functions = ["+", "-", "*", "%", "/"]

terminals_size = len(terminals) - 1
functions_size = len(functions) - 1


n_population = 10
n_iterations = 3

reproduction_prob = 0.1
mutation_prob = 0.1
crossover_prob = 0.8

individual_size = 7							#	(/(* x x)(/ 2 1))
function_pattern = [1, 1, 0, 0, 1, 0, 0]  	#	1 for function; 0 for terminal

data = []

def get_random_terminal():
	return random.randint(0, terminals_size)

def get_random_function():
	return random.randint(0, functions_size)

def function_fitness( x ):
	print(x)
	a = operators[functions[x[1]]] (terminals[x[2]], terminals[x[3]])
	b = operators[functions[x[4]]] (terminals[x[5]], terminals[x[6]])
	return operators[functions[x[0]]] (a, b)

def generate_population(n_population, individual_size):
	population = []
	for i in range(n_population):
		tmp_function = []
		for j in function_pattern:
			if( j ):
				tmp_function.append(get_random_function())
			else:
				tmp_function.append(get_random_terminal())
		population.append(tmp_function)
	print("Generating population:")
	print('\n'.join(' '.join(map(str,i)) for i in population))
	for i in population:
		data.append([i])

def eval_population():
	fitness = []
	for i in data:
		i.append([function_fitness( i[0] )])


def genetic_programing():

	generate_population(n_population, individual_size)
	eval_population()

	# for gen in range(n_iterations):



genetic_programing()