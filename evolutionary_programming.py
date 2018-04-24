import numpy as np
import random
import math
from operator import itemgetter

n_states = 4
n_char_states = 7
M = 100
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

def fitness_function( fsm_output ):
	misses = 0
	for i in range(len(fsm_output) -1 ):
		if( fsm_input[i+1] != fsm_output[i] ):
			misses += 1
	return misses

def generate_state():
	tmp_state = []
	for i in range(n_char_states):
		if( i < 5 ):
			tmp_state.append( random.randint(0,1) )
		else:
			tmp_state.append( random.randint(0, n_states-1))
	return tmp_state

def initialize_population():
	global data
	data = [ [[ generate_state() for i in range(n_states) ], \
				[ random.randint(0, n_states-1)], []] for i in range(population)]

def evaluate_population( db ):

	print("evaluate_population")

	for i in db:
		fsm_output = []
		fitness = np.inf
		current_state = i[1][0]
		c = 0
		for k in i[0]:
			if( k[0] == 1 ):
				c+=1

		if( i[0][current_state][0] == 1 and c>=2):

			for j in range( len( fsm_input ) ):
				if ( i[0][current_state][1] == fsm_input[j] and \
						i[0][i[0][current_state][5]][0] == 1 ):
					fsm_output.append( i[0][current_state][3] )
					current_state = i[0][current_state][5]
				elif ( i[0][current_state][2] == fsm_input[j] and \
						i[0][i[0][current_state][6]][0] == 1):
					fsm_output.append( i[0][current_state][4] )
					current_state = i[0][current_state][6]
				else:
					break

			if( len(fsm_input) == len(fsm_output) ):
				fitness = fitness_function( fsm_output )

		i[2]= fitness

def mutation( individual ):
	offspring = list(individual)
	mutation_prob = random.random()

	if( mutation_prob <= deactivate_state ):
		offspring[0][random.randint(0, n_states-1)][0] = 0
	elif( deactivate_state > mutation_prob <= change_initial_state ):
		offspring[1][0] = random.randint(0, n_states-1)
	elif( change_initial_state > mutation_prob <= change_input ):
		offspring[0][random.randint(0, n_states-1)][random.randint(1,2)] \
				= random.randint(0, 1)  
	elif( change_input > mutation_prob <= change_output ):
		offspring[0][random.randint(0, n_states-1)][random.randint(3,4)] \
				= random.randint(0, 1)  
	elif( change_output > mutation_prob <= active_state ):
		offspring[0][random.randint(0, n_states-1)][0] = 1

	return offspring

def ep_algorithm():
	global data
	iteration = 0

	initialize_population()
	

	while( True ):
		evaluate_population(data)
		offspring = []
		for i in data:
			tmp = i[:]
			offspring.append( mutation( tmp ) )
		evaluate_population(offspring)

		survivors = []
		tmp_1 = []
		tmp_2 = []

		for i,x in zip(data,range(len(data))):
			tmp_1.append([x, data[x][2] ] )
			tmp_2.append([x, offspring[x][2] ])
		tmp_1 = sorted(tmp_1, key=itemgetter(1), reverse=False)
		tmp_2 = sorted(tmp_2, key=itemgetter(1), reverse=False)

		print("data: ", tmp_1)
		print("Mutated ", tmp_2)

		for i in range( int(population / 2) ):
			survivors.append( data[tmp_1[i][0]] )
		for i in range( int(population / 2) ):
			survivors.append( offspring[tmp_2[i][0]] )
		
		if( data[tmp_1[i][0]][2] <= 2  ):
			print("FSM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			print( data[tmp_1[i][0]] )
			break
		if( offspring[tmp_2[i][0]][2] <= 2):
			print("FSM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			print(offspring[tmp_2[i][0]])
			break

		data = []
		data = survivors[:]
		iteration += 1

ep_algorithm()

print(data)