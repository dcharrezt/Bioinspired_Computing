import numpy as np
import random


distance_matrix = [ [0, 12, 3, 23, 1, 5, 23, 56, 12, 11, 89, 97, 52], 
					[12, 0, 9, 18, 3, 41, 45, 5, 41, 27, 16, 76, 56], 
					[3, 9, 0, 89, 56, 21, 12, 48, 14, 29, 5, 91, 8],
					[23, 18, 89, 0, 87, 46, 75, 17, 50, 42, 100, 70, 15], 
					[1, 3, 56, 87, 0, 55, 22, 86, 14, 33, 31, 84, 21],
					[5, 41, 21, 46, 55, 0, 21, 76, 54, 81, 92, 37, 22],
					[23, 45, 12, 75, 22, 21, 0, 11, 57, 48, 39, 59, 22],
					[56, 5, 48, 17, 86, 76, 11, 0, 63, 24, 55, 58, 98],
					[12, 41, 14, 50, 14, 54, 57, 63, 0, 9, 44, 18, 52],
					[11, 27, 29, 42, 33, 81, 48, 24, 9, 0, 64, 65, 82],
					[89, 16, 5, 100, 31, 92, 39, 55, 44, 64, 0, 9, 70],
					[97, 76, 91, 70, 84, 37, 59, 58, 18, 65, 9, 0, 50],
					[52, 56, 8, 15, 21, 22, 22, 98, 52, 82, 70, 50, 0] ]

# Parameters

p = 0.01
alpha = 1
beta = 2
q_0 = 0.9
phi = 0.1

initial_pheromones = 1.

n_ants = 10
n_iterations = 50
n_cities = 13

n_mutation = 10

first_city = random.randint(0, n_cities-1)

best_global = {'path':[], 'cost': np.inf}

pheromone_matrix = np.zeros(( n_cities, n_cities ))
visibility_matrix = np.zeros(( n_cities, n_cities ))

cities = [ 'A','B','C','D','E','F','G','H','I','J','K','L','M' ]

def print_matrix( matrix, text ):
	print( text )
	for i in range( n_cities ):
		if(i==0):
			print("\tA\tB\tC\tD\tE\tF\tG\tH\tI\tJ\tK\tL\tM")
		for j in range( n_cities ):
			if(j==0):
				print(cities[i], end='\t')
			print( "{:.3f}".format(matrix[i][j]), end='\t')
		print()

def initialize_pheromone_matrix():
	for i in range( n_cities ):
		for j in range( n_cities ):
			if(i!=j):
				pheromone_matrix[i][j] = initial_pheromones

def initialize_visibility_matrix():
	for i in range( n_cities ):
		for j in range( n_cities ):
			if i != j :
				visibility_matrix[i][j] = 1.0 / distance_matrix[i][j]

def next_city( m_prob, random_number):
	probabilty_sum = 0
	for i in range( len(m_prob) ):
		if( m_prob[i] != -1 ):
			probabilty_sum += m_prob[i]
			if( random_number <= probabilty_sum ):
				return i

def update_pheromone_trace( i, j):
	print("(1-"+str(phi)+")*"+str(pheromone_matrix[i][j])+"+"+str(phi)+"*"\
				+str(initial_pheromones)+"=",end='')
	pheromone_matrix[i][j] *= (1-phi) + phi*initial_pheromones
	print(pheromone_matrix[i][j])

def send_ants():
	path_list = []
	for i in range( n_ants ):
		path = []
		current_unit = random.randint(0, n_cities-1)
		path.append( current_unit )
		print("Ant # ",i )
		print("Starting at: ", current_unit)
		while( len(path) < n_cities ):
			m_sum = 0.
			sums_list = []
			n_index = -1
			q = random.random()
			print("Value of q: ", q)
			if q <= q_0 :
				print("Travel by diversification")

				for j in range( n_cities ):
					if j not in path :
						t = (pheromone_matrix[current_unit][j]) ** alpha
						n = (visibility_matrix[current_unit][j]) ** beta
						tn = t*n
						sums_list.append( tn )
						m_sum += tn
						print( cities[current_unit] + "-" + cities[j], end=' ' )
						print( "t = ", t, end=' ' )
						print( "n = ", n, end=' ' )
						print( "t*n = ", tn )
					else:
						sums_list.append( -1 )
				print( "Sum: ", m_sum )
				m_prob = []
				for k in range( n_cities ):
					if k not in path :
						m_prob.append( sums_list[k] / m_sum )
						print( cities[current_unit] + "-" + cities[k], end=' ' )
						print( "Probabilty = ", sums_list[k] / m_sum)
					else:
						m_prob.append(-1)
				random_number = random.random()
				print( "Random number: ", random_number )
				n_index = next_city( m_prob, random_number )
				print("Next city: ", cities[n_index] )

			else:
				print("Travel by Intensification")
				prev_tn = -1
				for j in range( n_cities ):
					if j not in path :
						t = (pheromone_matrix[current_unit][j]) ** alpha
						n = (visibility_matrix[current_unit][j]) ** beta
						tn = t*n
						sums_list.append( tn )
						m_sum += tn
						print( cities[current_unit] + "-" + cities[j], end=' ' )
						print( "t = ", t, end=' ' )
						print( "n = ", n, end=' ' )
						print( "t*n = ", tn )
						if prev_tn < tn :
							n_index = j
				print("Next city: ", cities[n_index] )

			print("Updating arc "+str(cities[path[-1]])+"-"+str(cities[n_index])\
												+": ",end='')
			update_pheromone_trace(path[-1], n_index)
			current_unit = n_index
			path.append( n_index )
		print("Ant # "+str(i)+": ", end='')
		for i in range( n_cities ):
			if( i == n_cities-1 ):
				print( cities[path[i]])
			else:	
				print( cities[path[i]] + "-", end='')
		path_list.append( path )
	return path_list

def path_cost( path ):
	m_sum = 0.
	for i in range( len( path )-1 ):
		m_sum += distance_matrix[path[i]][path[i+1]]
	return m_sum

def print_ant_results( path_list ):
	print("\nResults")
	costs_lists = []
	for j in range( len( path_list ) ):
		print("Ant # "+str(j)+": ", end='') 
		for i in range( n_cities ):
			if( i == n_cities-1 ):
				print( cities[path_list[j][i]], end=' ')
			else:
				print( cities[ path_list[j][i]] + "-", end='')
		costs_lists.append( path_cost(path_list[j]) ) 
		print( "Cost: ", costs_lists[j])
	index_ant = costs_lists.index( min(costs_lists) )
	print("------------------------------------------------------")
	print("Best Local Ant: ", end='')
	for i in range( n_cities ):
		if( i == n_cities-1 ):
			print( cities[path_list[index_ant][i]], end=' ')
		else:
			print( cities[path_list[index_ant][i]] + "-", end='')
	print("Cost: ", costs_lists[index_ant])
	print("------------------------------------------------------")

	if best_global['cost'] > costs_lists[index_ant] :
		best_global['path'] = list(path_list[index_ant])
		best_global['cost'] = costs_lists[index_ant]

	print("------------------------------------------------------")
	print("Best Global Ant: ", end='')
	for i in range( n_cities ):
		if( i == n_cities-1 ):
			print( cities[best_global['path'][i]], end=' ' )
		else:
			print( cities[best_global['path'][i]] + "-", end='' )
	print("Cost: ", best_global['cost'])
	print("------------------------------------------------------")

	return costs_lists

def get_delta( i, j):
	for k in range( len(best_global['path']) -1 ):
		if ( best_global['path'][k] == i and best_global['path'][k+1] == j) or \
			( best_global['path'][k+1] == i and best_global['path'][k] == j):
			return 1 / best_global['cost']
	return 0

def update_pheromone_matrix( path_list, costs_lists ):
	for i in range( n_cities ):
		for j in range( n_cities ):
			tmp = 0
			if i != j :
				print(cities[i]+" "+cities[j]+": Pheromone = ", end='')
				delta = get_delta(i, j)
				print( str(1.-p)+"*"+str(pheromone_matrix[i][j])+\
									"+"+str(delta)+" = ", end='')
				pheromone_matrix[i][j] *= (1-p)+p*delta
				print( pheromone_matrix[i][j] )

def first_option_mutation( path ):
	tmp_path = path[:]
	best_path = path[:]
	while(True):
		best_cost = path_cost(best_path)
		better_than_best = False
		for i in range( n_mutation ):
			rand_1 = random.randint(0, n_cities-1)
			rand_2 = random.randint(0, n_cities-1)
			tmp_path[rand_1], tmp_path[rand_2] = tmp_path[rand_2], tmp_path[rand_1]
			tmp_cost = path_cost( tmp_path )
			if tmp_cost < best_cost :
				best_path = tmp_path[:]
				better_than_best = True
				break
		if not better_than_best :
			return best_path, best_cost

def local_search( path_list, costs_lists):
	paths_mutated = []
	costs_mutated = []
	for i in range( len(path_list) ):
		best_path, best_cost = first_option_mutation( path_list[i] )
		if best_path != path_list[i] :
			paths_mutated.append( best_path )
			costs_mutated.append( best_cost )
	path_list += paths_mutated
	costs_lists += costs_mutated 

def ACS_algorithm():
	initialize_pheromone_matrix()
	initialize_visibility_matrix()

	for i in range( n_iterations ):
		print("Iteration # ", i)
		if(i == 0):
			print_matrix( distance_matrix, " Distance Matrix " )
			print_matrix( pheromone_matrix, " Pheromone Matrix" )
			print_matrix( visibility_matrix, "Visibility Matrix" )
		path_list = send_ants()
		cost_list = print_ant_results( path_list )
		local_search( path_list, cost_list )
		update_pheromone_matrix( path_list, cost_list )

	print_matrix( pheromone_matrix, " Updated Pheromone Matrix " )


if __name__ == "__main__":

	ACS_algorithm()
