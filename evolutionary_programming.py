import numpy as np
import random

n_states = 4
n_char_states = 7
M = 4
min_active_states = 2   # admitted in the population
population = M
n_iterations = 100

fsm_input = [0,1,1,0,1,1,0,1,1,0,1,1,0,1,1]

# Mutation probabilities
deactivate_state = 0.2
change_initial_state = 0.4
change_input = 0.6
change_output = 0.8
active_state = 1.0

data = []

def fitness( fsm_output ):
	matches = 0
	for i in range(len(fsm_output) -1 ):
		if( fsm_input[i+1] == fsm_output[i] ):
			matches += 1
	return matches

def generate_state():
	tmp_state = []
	for i in range(n_char_states):
		if( i < 5 ):
			tmp_state.append( random.randint(0,1) )
		else:
			tmp_state.append( random.randint(0, n_states-1))
	return tmp_state

def initialize_population():
	data = [ [[ generate_state() for i in range(n_states) ], \
				[ random.randint(0, n_states-1)]] for i in range(population)]

	# print(data)

# def evuluate_population():
# 	for i in data:
# 		fsm_output = []
# 		initial_state = i[1]
# 		for j in mef_input:
			



initialize_population()