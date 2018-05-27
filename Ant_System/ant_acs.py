


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
n_iterations = 0.

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


def ACS_algorithm():
	iteration = 0
	for i in range( n_iterations ):
		print("Iteration #", i)



if __name__ == "__main__":

	print("Hi") 
