
import numpy as np
import random

# Parameters 

distances = [ [0, 1, 3, 4, 5],
			  [1, 0, 1, 4, 8],
			  [3, 1, 0, 5, 1],
			  [4, 4, 5, 0, 2],
			  [5, 8, 1, 2, 0] ]

n_iterations = 4
n_particles = 4
n_cities = 5

phi_1 = 1.0
phi_2 = 1.0

data = []
cities = [ 'A', 'B', 'C', 'D', 'E' ]
best_global = {'path': [], 'fitness': np.inf }

def create_particle():
	path = np.random.permutation( n_cities )
	return {'path': path, 'fitness':np.inf, 'best_local_path':[], 
						'b_local_fitness': np.inf, 'vel':[] }

def create_swarm():
	for i in range( n_particles ):
		data.append( create_particle() )

def function_fitness( path ):
	cost = 0.
	for i in range( n_cities-1 ):
		cost += distances[ path[i] ][ path[i+1] ]
	return cost

def evaluating_swarm():
	for i in range( n_particles ):
		data[i]['fitness'] = function_fitness( data[i]['path'] )
		if data[i]['fitness'] < best_global['fitness']:
			best_global['fitness'] = data[i]['fitness']
			best_global['path'] = list(data[i]['path'])
		if data[i]['fitness'] < data[i]['b_local_fitness']:
			data[i]['b_local_fitness'] = data[i]['fitness']
			data[i]['best_local_path'] = data[i]['path']

def substract_permutations( path_1, path_2 ):
	SS = []
	tmp_1 = list( path_1 )
	tmp_2 = list( path_2 )
	for i in range( n_cities ):
		if tmp_1[i] != tmp_2[i]:
			index = tmp_2.index( tmp_1[i] )
			SS.append([i , index])
			tmp = tmp_2[i]
			tmp_2[i] = tmp_2[index]
			tmp_2[index] = tmp 
	return SS
	
def update_particle( particle ):
	print( "1 ", particle['path'] )
	for i in particle['vel']:
		tmp = particle['path'][i[0]]
		particle['path'][i[0]] = particle['path'][i[1]]
		particle['path'][i[1]] = tmp
	particle['vel'] = []
	print( "2 ", particle['path'] )


def pso():
	print("****** Starting PSO")
	create_swarm()
	for i in range( n_iterations ):
		print( "Iteration " + str(i+1) + " ******************* ")
		evaluating_swarm()
		for i in data:
			print("Current: ", i['path'])
			print("Fitness: ", i['fitness'] )
			print("Best local", i['best_local_path'] )
			print("Fitness Best local", i['b_local_fitness'])
			print("Velocity: ", i['vel'] , end='\n\n' )
			a = substract_permutations( i['path'], best_global['path'])
			b = substract_permutations( i['path'], i['best_local_path'])
			i['vel'] = list(a + b)
			update_particle( i )
		print("Best Global *** ")
		print("Path: ", best_global['path'])
		print("Fitness: ", best_global['fitness'])
		print("")

if __name__=="__main__":
	pso()