import numpy as np
import random
import copy
import matplotlib.pyplot as plt

# parameters

n_iterations = 5
n_particles = 4
n_dimesions = 2

min_v = -1
max_v = 1

min_x = 0
max_x = 5

min_y = 0
max_y = 3

phi_1 = 2.0
phi_2 = 2.0

data = []
global_repository = []

def function_1( x, y ):
	return 4*(x**2) + 4*(y**2)

def function_2( x, y ):
	return (x-5)**2 + (y-5)**2

def create_particle():
	x = random.uniform(min_x, max_x)
	y = random.uniform(min_y, max_y)
	v_x = random.uniform(min_v, max_v)
	v_y = random.uniform(min_v, max_v)
	fitness = np.inf
	return { "pos": [x, y], "vel": [v_x, v_y], "fitness_1": fitness,\
						"fitness_2": fitness, "repo": [] } 

def create_swarm():
	for i in range( n_particles ):
		data.append( create_particle() )

def dominate( particle_1, particle_2 ):
	if( ( particle_1["fitness_1"] < particle_2["fitness_1"] and \
	      particle_1["fitness_2"] < particle_2["fitness_2"] ) or \
	    ( particle_1["fitness_1"] <= particle_2["fitness_1"] and \
	   	  particle_1["fitness_2"] < particle_2["fitness_2"] ) or \
	    ( particle_1["fitness_1"] < particle_2["fitness_1"] and \
	   	  particle_1["fitness_2"] <= particle_2["fitness_2"] ) ) :
		return True
	return False

def evaluating_swarm():
	global data
	indexes_to_delete = []
	for i in range( len(data) ):
		if  ( data[i]["pos"][0] < min_x or data[i]["pos"][0] > max_x ) or \
			( data[i]["pos"][1] < min_y or data[i]["pos"][1] > max_y ):
			indexes_to_delete.append( i )
		else:
			data[i]["fitness_1"] = function_1( data[i]["pos"][0], data[i]["pos"][1] )
			data[i]["fitness_2"] = function_2( data[i]["pos"][0], data[i]["pos"][1] )
	tmp_data = []
	for i in range( len( data ) ):
		if i not in indexes_to_delete:
			tmp_data.append( copy.deepcopy( data[i] ) )
	data = []
	data = copy.deepcopy( tmp_data )

def print_swarm():
	for i in range(len(data)):
		print( "particle #" + str(i) +
			   " x = "+str(data[i]["pos"][0])+
			   " y = "+str(data[i]["pos"][1])+
			   " v_x = "+str(data[i]["vel"][0])+
			   " v_y = "+str(data[i]["vel"][1]) )

def non_dominated_sort( from_data ):
	S = []
	N = []
	rank = []
	frontiers = [[]]
	
	for i in range( len( from_data ) ):
		S.append([])
		N.append( 0 )
		rank.append( 0 )

	for p in  range(len( from_data )) :
		for q in range(len( from_data )):
			if( dominate(from_data[p], from_data[q]) ):
				S[p].append(q)
			elif( dominate( from_data[q], from_data[p]) ):
				N[p] += 1
		if(N[p] == 0):
			rank[p] = 0
			frontiers[0].append(p)
	i = 0
	while( frontiers[i] != [] ):
		Q = []
		for p in frontiers[i]:
			for q in S[p]:
				N[q] -= 1
				if( N[q] == 0 ):
					rank[q] = i+1
					Q.append(q)
		i += 1
		frontiers.append( Q )
	del frontiers[len(frontiers)-1]
	return frontiers

def update_global_repository( pareto_front, swarm ):
	global global_repository
	global_repository = []
	# indexes_to_delete = []
	for i in range( len(swarm) ):
		if i in pareto_front:
			global_repository.append( copy.deepcopy( swarm[i] ) )

	# for i in pareto_front:
	# 	if (( data[i]["pos"][0] < min_x or data[i]["pos"][0] > max_x ) or \
	# 		( data[i]["pos"][1] < min_y or data[i]["pos"][1] > max_y )):
	# 		print("deleted out range")
	# 		indexes_to_delete.append( i )
	
	# for i in range( global_repository ):



def update_local_repository():
	for i in range( len(data) ):
		data[i]["repo"].append( copy.deepcopy(data[i]) )

def best_local_particle( particle ):
	print("PARTICLEEE ", particle)
	rand = random.randint(0, len(particle["repo"]) - 1)
	return particle["repo"][rand]

def best_global_particle():
	rand = random.randint(0, len( global_repository ) -1 )
	return global_repository[ rand ]

def updating_position( particle, pLocal, pGlobal ):
	for i in range( n_dimesions ):
		w = random.random()
		rand_1 = random.random()
		rand_2 = random.random()
		V = w*particle["pos"][i] + phi_1*rand_1*(pLocal["pos"][i]-\
			particle["pos"][i] ) + phi_2*rand_2*(pGlobal["pos"][i]- \
			particle["pos"][i])
		particle["pos"][i] += V

def mopso():
	print("****** Starting PSO")
	create_swarm()
	print_swarm()
	evaluating_swarm()

	pareto_front = non_dominated_sort( data )
	update_global_repository( pareto_front[0], data )
	update_local_repository()


	for i in range( n_iterations ):
		print("Iteration: ", i )
		for i in range( len(data) ):
			pLocal = best_local_particle( copy.deepcopy(data[i]) )
			pGlocal = best_global_particle()
			updating_position(data[i], pLocal, pGlocal)
		evaluating_swarm()
		new_swarm = data + global_repository
		pareto_front = non_dominated_sort( new_swarm )
		update_global_repository( pareto_front[0], new_swarm )
		# update_local_repository()
	for i in range(len(global_repository)):
		print( "particle #" + str(i) +
			   "\nx = "+str(global_repository[i]["pos"][0])+
			   "\ny = "+str(global_repository[i]["pos"][1])+
			   "\nv_x = "+str(global_repository[i]["vel"][0])+
			   "\nv_y = "+str(global_repository[i]["vel"][1])+
			   "\nfitness_1 = "+str(global_repository[i]["fitness_1"] )+
			   "\nfitness_2 = "+str(global_repository[i]["fitness_2"] ))

if __name__=="__main__":
	mopso()

	plt.plot([ i["fitness_1"] for i in global_repository ], \
				[i["fitness_2"] for i in global_repository], 'ro')
	# plt.axis([0, 6, 0, 20])
	plt.show()



