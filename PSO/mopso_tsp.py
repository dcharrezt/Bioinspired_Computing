import random
import copy
import numpy as np
import matplotlib.pyplot as plt

city_distances = [ [0, 12, 3, 23, 1, 5, 23, 56, 12, 11],
				   [12, 0, 9, 18, 3, 41, 45, 5, 41, 27],
				   [3, 9, 0, 89, 56, 21, 12, 48, 14, 29],
				   [23, 18, 89, 0, 87, 46, 75, 17, 50, 42],
				   [1, 3, 56, 87, 0, 55, 22, 86, 14, 33],
				   [5, 41, 21, 46, 55, 0, 21, 76, 54, 81],
				   [23, 45, 12, 75, 22, 21, 0, 11, 57, 48],
				   [56, 5, 48, 17, 86, 76, 11, 0, 63, 24],
				   [12, 41, 14, 50, 14, 54, 57, 63, 0, 9],
				   [11, 27, 29, 42, 33, 81, 48, 24, 9, 0] ]

cost_between_cities = [ [0, 22, 47, 15, 63, 21, 23, 16, 11, 9], 
						[22, 0, 18, 62, 41, 52, 13, 11, 26, 43],
						[47, 18, 0, 32, 57, 44, 62, 20, 8, 36],
						[15, 62, 32, 0, 62, 45, 75, 63, 14, 12],
						[63, 41, 57, 62, 0, 9, 99, 42, 56, 23],
						[21, 52, 44, 45, 9, 0, 77 ,58, 22, 14],
						[23, 13, 62, 75, 99, 77, 0, 30, 25, 60],
						[16, 11, 20, 63, 42, 58, 30, 0, 66, 85],
						[11, 26, 8, 14, 56, 22, 25, 66, 0, 54],
						[9, 43, 36, 12, 23, 14, 60, 85, 54, 0]]

n_iterations = 3
n_particles = 20
n_dimesions = 2
n_fitness_functions = 2
n_cities = 10

phi_1 = 1.0
phi_2 = 1.0

class Particle:
	def __init__(self, path):
		self.path = path[:]
		self.velocity = []
		self.f_distance = np.inf
		self.f_cost = np.inf
		self.local_repository = []

data = []
global_repository = []

def fitness_distance( path ):
	distance = 0.
	for i in range( n_cities - 1 ):
		distance += city_distances[ path[i] ][ path[i+1] ]
	return distance

def fitness_cost( path ):
	cost = 0.
	for i in range( n_cities - 1 ):
		cost += cost_between_cities[path[i]][path[i+1]]
	return cost

def create_swarm():
	for i in range( n_particles ):
		path = np.random.permutation( n_cities )
		tmp = Particle(path)
		tmp.local_repository.append( copy.deepcopy(tmp) )
		data.append( copy.deepcopy(tmp) ) 

def evaluate_swarm():
	for i in data:
		i.f_distance = fitness_distance( i.path )
		i.f_cost = fitness_cost( i.path )

def dominate( particle_1, particle_2 ):
	if( ( particle_1.f_distance < particle_2.f_distance  and \
		  particle_1.f_cost < particle_2.f_cost ) or \
		( particle_1.f_distance <= particle_2.f_distance and \
		  particle_1.f_cost < particle_2.f_cost ) or \
		( particle_1.f_distance < particle_2.f_distance and \
		  particle_1.f_cost <= particle_2.f_cost ) ):
		return True
	return False

def update_global_repository():
	global global_repository
	if len( global_repository ) == 0:
		global_repository.append( copy.deepcopy(data[0]) )

	for i in data:
		for j in global_repository:
			

def update_local_repository():
	for i in data:
		non_dominated_indexes = []
		new_pareto = []
		for j in range( len(i.local_repository) ):
			if not ( dominate( i, i.local_repository[j] ) ):
				non_dominated_indexes.append( j )
		if len( non_dominated_indexes ) != len( i.local_repository ):
			for k in non_dominated_indexes:
				new_pareto.append( i.local_repository[k] )
			new_pareto.append( copy.deepcopy( i ) )
			i.local_repository = copy.deepcopy( new_pareto )

def mopso_tsp():
	create_swarm()
	evaluate_swarm()
	update_global_repository()
	update_local_repository()
	for i in range( n_iterations ):
		print( "+++++ Iteration ", i )



if __name__=="__main__":
	mopso_tsp()


	for i in data:
		print(i.path)
		print(i.f_distance)
		print(i.f_cost)

	print("GLObal")
	for m in global_repository:
		print(m.path)
		print(m.f_distance)
		print(m.f_cost)
	print()

	plt.plot([ i.f_distance for i in global_repository ], \
				[i.f_cost for i in global_repository], 'ro')
	# plt.axis([0, 6, 0, 20])
	plt.show()

	# print("GLObal")

	# for i in global_repository:
	# 	print(i.path)
	# 	print(i.f_distance)
	# 	print(i.f_cost)

	# pos = [1, 2]
	# a = Particle( pos,[-1,1],[2,6] )
	# pos += [3, 4]
	# b = Particle( pos,[-1,1],[2,6] )
	# a.local_repository.append( copy.deepcopy(b) )
	# b.position = [5,6]  

	# print(a.local_repository[0].position)
	# print(b.position)