import numpy as np
import random


distance_matrix = [ [0, 12, 3, 23, 1, 5, 23, 56, 12, 11],
				   [12, 0, 9, 18, 3, 41, 45, 5, 41, 27],
				   [3, 9, 0, 89, 56, 21, 12, 48, 14, 29],
				   [23, 18, 89, 0, 87, 46, 75, 17, 50, 42],
				   [1, 3, 56, 87, 0, 55, 22, 86, 14, 33],
				   [5, 41, 21, 46, 55, 0, 21, 76, 54, 81],
				   [23, 45, 12, 75, 22, 21, 0, 11, 57, 48],
				   [56, 5, 48, 17, 86, 76, 11, 0, 63, 24],
				   [12, 41, 14, 50, 14, 54, 57, 63, 0, 9],
				   [11, 27, 29, 42, 33, 81, 48, 24, 9, 0] ]

# distance_matrix = [ [0, 12, 3, 23, 1],
# 				   [12, 0, 9, 18, 3],
# 				   [3, 9, 0, 89, 56],
# 				   [23, 18, 89, 0, 87],
# 				   [1, 3, 56, 87, 0] ]

first_city = 3
p = 0.99
alpha = 1
beta = 1
Q = 1
initial_pheromones = .1

n_ants = 3
n_iterations = 100
n_cities = 5

e = 5
w = 6

pheromone_matrix = np.zeros(( n_cities, n_cities ))
visibility_matrix = np.zeros(( n_cities, n_cities ))


cities = [ 'A','B','C','D','E','F','G','H','I','J' ]

def print_matrix( matrix, text ):
	print( text )
	for i in range( n_cities ):
		if(i==0):
			print("\tA\tB\tC\tD\tE\tF\tG\tH\tI\tJ" )
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
			if(i!=j):
				visibility_matrix[i][j] = 1.0 / distance_matrix[i][j]

def next_city( m_prob, random_number):
	probabilty_sum = 0
	for i in range( len(m_prob) ):
		if( m_prob[i] != -1 ):
			probabilty_sum += m_prob[i]
			if( random_number <= probabilty_sum ):
				return i

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
	print("Best Ant: ", end='')
	for i in range( n_cities ):
		if( i == n_cities-1 ):
			print( cities[path_list[index_ant][i]], end=' ')
		else:
			print( cities[ path_list[index_ant][i]] + "-", end='')
	print("Cost: ", costs_lists[index_ant])
	print("------------------------------------------------------")

	return costs_lists

def print_ant_results_rank( path_list ):
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
	print("Best Ant: ", end='')
	for i in range( n_cities ):
		if( i == n_cities-1 ):
			print( cities[path_list[index_ant][i]], end=' ')
		else:
			print( cities[ path_list[index_ant][i]] + "-", end='')
	print("Cost: ", costs_lists[index_ant])
	print("------------------------------------------------------")

	ranking = rank_ants( path_list, costs_lists )

	return ranking

def send_ants():
	global first_city
	path_list = []

	for j in range( n_ants ):
		print("Ant # ", j)
		print("Initial city: ", cities[first_city])
		current_city = first_city
		path = []
		path.append( current_city )
		while( len(path) < n_cities ):
			m_sum = 0
			sums_list = []
			for k in range( n_cities ):
				if( k not in path ):
					t = (pheromone_matrix[current_city][k]) ** alpha
					n = (visibility_matrix[current_city][k]) ** beta
					tn = t*n
					sums_list.append( tn )
					m_sum += tn
					print( cities[current_city] + "-" + cities[k], end=' ' )
					print( "t = ", t, end=' ' )
					print( "n = ", n, end=' ' )
					print( "t*n = ", tn )
				else:
					sums_list.append(-1)
			m_prob = []
			for k in range( n_cities ):
				if( k not in path ):
					m_prob.append( sums_list[k] / m_sum )
					print( cities[current_city] + "-" + cities[k], end=' ' )
					print( "Probabilty = ", sums_list[k] / m_sum)
				else:
					m_prob.append(-1)
			random_number = random.random()
			print( "Random number: ", random_number )
			n_index = next_city( m_prob, random_number )
			print("Next city: ", cities[n_index] )
			current_city = n_index
			path.append( n_index )
		print("Ant # "+str(j)+": ", end='')
		for i in range( n_cities ):
			if( i == n_cities-1 ):
				print( cities[path[i]])
			else:	
				print( cities[path[i]] + "-", end='')

		path_list.append( path )
	return path_list

def get_delta(path_list, costs_lists, i , j):
	# print( i, "\t", j )
	s = 0
	for k in range( len( path_list )):
		for l in range( n_cities -1 ):
			if( (path_list[k][l] == i and path_list[k][l+1] == j) or \
				(path_list[k][l] == j and path_list[k][l+1] == i) ):
				# print( "ms ",  Q / costs_lists[k])
				s += Q / costs_lists[k]
	# print("s ", s)
	return s

def update_pheromone_matrix(path_list, costs_lists):
	global p
	for r_0 in range( n_cities ):
		for r_1 in range( n_cities ):
			if( r_0 != r_1):
				tmp = get_delta(path_list, costs_lists, r_0, r_1)
				pheromone_matrix[r_0][r_1] *= p + tmp

def check_best_rank( path, cost, i, j):
	for k in range( len(path) - 1 ):
		if( ( path[k]== i and path[k+1] == j) or \
				(path[k] == j and path[k+1] == i) ):
			return e * ( 1 / cost )
	return 0

def rank_ants( path_list, costs_lists):
	ranking = []
	for i in range( len( costs_lists )):
		ranking.append( [i, costs_lists[i], path_list[i] ] )

	# print( ranking )

	ranking = sorted(ranking, key=lambda x : x[1] )

	print( " --- Ranking --- ")
	for j in range( len( ranking ) ):
		print("Ant # "+str(j)+": ", end='') 
		for i in range( n_cities ):
			if( i == n_cities-1 ):
				print( cities[ranking[j][2][i]], end=' ')
			else:
				print( cities[ ranking[j][2][i]] + "-", end='')
		print( "Cost: ", ranking[j][1])

	return ranking

def get_delta_rank( ranking, i , j):
	# print( i, "\t", j )
	s = 0
	for ms in ranking: 
		for k in range( n_cities - 1 ):
			if( (ms[2][k] == i and ms[2][k+1] == j)  or \
					ms[2][k] == j and ms[2][k+1] == i ):
				s += ( w-ms[0] ) * Q / ms[1]
		if ms[0] == 0:
			s+= w * ( 1 / ms[1])
	return s

def update_pheromone_elitist( path_list, costs_lists ):
	index_ant = costs_lists.index( min(costs_lists) )
	global p
	for r_0 in range( n_cities ):
		for r_1 in range( n_cities ):
			if( r_0 != r_1):
				tmp = get_delta(path_list, costs_lists, r_0, r_1)
				best = check_best_rank( path_list[index_ant], \
											costs_lists[index_ant], r_0, r_1 )

				pheromone_matrix[r_0][r_1] *= p + tmp + best
				# print(r_0 + " " r_1 " = " )

def update_pheromone_rankings( ranking ):
	global p
	for r_0 in range( n_cities ):
		for r_1 in range( n_cities ):
			if( r_0 != r_1):
				tmp = get_delta_rank(ranking, r_0, r_1)
				pheromone_matrix[r_0][r_1] *= p + tmp
		# if( ranking[] == 0):

def as_algorithm():
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
		# ranking = print_ant_results_rank( path_list )
		# print(ranking)

		update_pheromone_matrix( path_list, cost_list )
		# update_pheromone_elitist( path_list, cost_list)
		# update_pheromone_rankings( ranking )

	print_matrix( pheromone_matrix, " Updated Pheromone Matrix ")


if __name__ == "__main__":
	as_algorithm()