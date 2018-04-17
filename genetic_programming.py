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
outputs = [0, 0.005, 0.02, 0.045, 0.08, 0.125, 0.18, 0.245, 0.32, 0.405]

terminals = [-54, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
functions = ["+", "-", "*", "%", "/"]

terminals_size = len(terminals) - 1
functions_size = len(functions) - 1

n_population = 5
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
	MSE = 0
	for i in range(len(inputs)):
		# 2 3 5 6
		c_1 = inputs[i] if terminals[x[2]]==-54 else terminals[x[2]]
		c_2 = inputs[i] if terminals[x[3]]==-54 else terminals[x[3]]
		c_3 = inputs[i] if terminals[x[5]]==-54 else terminals[x[5]]
		c_4 = inputs[i] if terminals[x[6]]==-54 else terminals[x[6]]

		if( (functions[x[1]] == "/" or functions[x[1]] == "%") and c_2 == 0):
			a = 0
		else:
			a = operators[functions[x[1]]] ( c_1, c_2)

		if( (functions[x[4]] == "/" or functions[x[4]] == "%") and c_4 == 0):
			b = 0
		else:
			b = operators[functions[x[4]]] ( c_3, c_4)
		
		if( (functions[x[0]] == "/" or functions[x[0]] == "%") and b == 0):
			c = 0
		else:
			c = operators[functions[x[0]]] (a, b)
		MSE += ( outputs[i] - c )**2
	return MSE / len(inputs)

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
	# print('\n'.join(' '.join(map(str,i)) for i in population))
	
	for i in population:
		data.append([i, [0]])
	print_symbols()

def eval_population():
	fitness = []
	for i in data:
		i[1][0] = ([function_fitness( i[0] )])

def print_symbols():

	for i in data:
		for j, k in zip(i[0], function_pattern):
			if( k == 1 ):
				print(functions[j], end='  ')
			else:
				if(terminals[j] == -54):
					print('x', end='  ')
				else:
					print(terminals[j], end='  ')
		print("\t", i[1][0])

def genetic_programing():

	generate_population(n_population, individual_size)
	eval_population()

	for i in range(n_iterations):
		print("\n\nIteration " + str(i) + " :\n\n")
		print("Evaluating individuals:")
		# print('\n'.join('\t'.join(' '.join(map(str, j))for j in i)for i in data ))
		print_symbols()


genetic_programing()