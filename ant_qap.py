import random
import numpy as np

distance_matrix = [ [0, 12, 6, 4],
					[12, 0, 6, 8],
					[6, 6, 0, 7],
					[4, 8, 7, 0] ]

flow_matrix = [ [0, 3, 8, 3],
				[3, 0, 2, 4],
				[8, 2, 0, 5],
				[3, 4, 5, 0] ]

n_ants = 4
n_units = 4
initial_pheromones = 1.

alpha = 1
beta = 1

min_pheromone = 0.1
max_pheromone = 1.

n_iterations = 10

pheromone_matrix = np.zeros(( n_units, n_units ))
visibility_matrix = np.zeros(( n_units, n_units ))

units = ['A', 'B', 'C', 'D']

def print_matrix( matrix, text ):
	print( text )
	for i in range( n_units ):
		if(i==0):
			print("\tA\tB\tC\tD" )
		for j in range( n_units ):
			if(j==0):
				print(units[i], end='\t')
			print( "{:.3f}".format(matrix[i][j]), end='\t')
		print()

def initialize_pheromone_matrix():
	for i in range( n_units ):
		for j in range( n_units ):
			if(i!=j):
				pheromone_matrix[i][j] = initial_pheromones

def initialize_visibility_matrix():
	for i in range( n_units ):
		for j in range( n_units ):
			if(i!=j):
				visibility_matrix[i][j] = 1.0 / \
								( distance_matrix[i][j] * flow_matrix[i][j] )

def next_city( m_prob, random_number):
	probabilty_sum = 0
	for i in range( len(m_prob) ):
		if( m_prob[i] != -1 ):
			probabilty_sum += m_prob[i]
			if( random_number <= probabilty_sum ):
				return i

def send_ants():
	path_list = []
	for i in range( n_ants ):
		path = []
		current_unit = random.randint( 0, n_units-1 )
		path.append( current_unit )
		print("Ant # ",i )
		print("Starting at: ", current_unit)
		while( len(path) < n_units ):
			m_sum = 0.
			sums_list = []
			for j in range( n_units ):
				if j not in path :
					t = (pheromone_matrix[current_unit][j]) ** alpha
					n = (visibility_matrix[current_unit][j]) ** beta
					tn = t*n
					sums_list.append( tn )
					m_sum += tn
					print( units[current_unit] + "-" + units[j], end=' ' )
					print( "t = ", t, end=' ' )
					print( "n = ", n, end=' ' )
					print( "t*n = ", tn )
				else:
					sums_list.append( -1 )
			print( "Sum: ", m_sum )
			m_prob = []
			for k in range( n_units ):
				if k not in path :
					m_prob.append( sums_list[k] / m_sum )
					print( units[current_unit] + "-" + units[k], end=' ' )
					print( "Probabilty = ", sums_list[k] / m_sum)
				else:
					m_prob.append(-1)
			random_number = random.random()
			print( "Random number: ", random_number )
			n_index = next_city( m_prob, random_number )
			print("Next city: ", units[n_index] )
			current_unit = n_index
			path.append( n_index )
		print("Ant # "+str(i)+": ", end='')
		for i in range( n_units ):
			if( i == n_units-1 ):
				print( units[path[i]])
			else:	
				print( units[path[i]] + "-", end='')
		path_list.append( path )
	return path_list

def min_max_algorithm():

	initialize_visibility_matrix()
	initialize_pheromone_matrix()

	path_list = send_ants()

	# for i in range( n_iterations ):
	# 	print

if __name__ == "__main__":

	min_max_algorithm()