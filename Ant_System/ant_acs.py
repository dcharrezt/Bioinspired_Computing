


distance_matrix = [ [0, 50, 50, 94, 50],
					[50, 0, 22, 50, 36],
					[50, 22, 0, 44, 14],
					[94, 50, 44, 0 ,50],
					[50, 36, 14, 50, 0] ]

flow_matrix = [ [0, 0, 2, 0, 3],
			    [0, 0, 0, 3, 0],
			    [2, 0, 0, 0, 0],
			    [0, 3, 0, 0, 1],
			    [3, 0, 0, 1, 0] ]

n_ants = 4
n_units = 5
initial_pheromones = 1.

alpha = 1
beta = 1
p = 0.01

min_pheromone = 0.1
max_pheromone = 1.

n_iterations = 100

pheromone_matrix = np.zeros(( n_units, n_units ))
visibility_matrix = np.zeros(( n_units, n_units ))

units = ['A', 'B', 'C', 'D', 'E']

def print_matrix( matrix, text ):
	print( text )
	for i in range( n_units ):
		if(i==0):
			print("\tA\tB\tC\tD\tE" )
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


def ACS_algorithm():
	iteration = 0
	for i in range( n_iterations ):
		print("Iteration #", i)



if __name__ == "__main__":

	print("Hi") 
