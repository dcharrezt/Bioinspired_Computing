import numpy as np
import random
import copy


# parameters

n_iterations = 2
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
	return 4*x**2 + 4*x**2

def function_2( x, y):
	return (x-5)**2 + (y-5)**2

def create_particle():
	x = random.uniform(min_x, max_x)
	y = random.uniform(min_y, max_y)
	v_x = random.uniform(-1, 1)
	v_y = random.uniform(-1, 1)
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
	for part, i in zip(data, range( len(data))):
		if( part["pos"][0] < min_x or part["pos"][0] > max_x ) or \
			( part["pos"][1] < min_y or part["pos"][1] > max_y ):
			del data[i]
		else:
			part["fitness_1"] = function_1( part["pos"][0], part["pos"][1] )
			part["fitness_2"] = function_2( part["pos"][0], part["pos"][1] )

def print_swarm():
	for i in range(len(data)):
		print( "particle #" + str(i) +
			   " x = "+str(data[i]["pos"][0])+
			   " y = "+str(data[i]["pos"][1])+
			   " v_x = "+str(data[i]["vel"][0])+
			   " v_y = "+str(data[i]["vel"][1]) )

def non_dominated_sort():
	S = []
	N = []
	rank = []
	frontiers = [[]]
	
	for i in range( len( data ) ):
		S.append([])
		N.append( 0 )
		rank.append( 0 )

	for p in  range(len( data )) :
		for q in range(len( data )):
			if( dominate(data[p], data[q]) ):
				S[p].append(q)
			elif( dominate( data[q], data[p]) ):
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

def update_global_repository( pareto_front ):
	global global_repository
	for i in pareto_front:
		if (( data[i]["pos"][0] < min_x or data[i]["pos"][0] > max_x ) or \
			( data[i]["pos"][1] < min_y or data[i]["pos"][1] > max_y )):
			print("deleted out range")
			del global_repository[i]
		else:
			global_repository.append( copy.deepcopy(data[i]) )

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

	pareto_front = non_dominated_sort()
	update_global_repository( pareto_front[0] )
	update_local_repository()


	for i in range( n_iterations ):
		print("Iteration: ", i )
		for i in range( len(data) ):
			pLocal = best_local_particle( copy.deepcopy(data[i]) )
			pGlocal = best_global_particle()
			updating_position(data[i], pLocal, pGlocal)
		evaluating_swarm()
		pareto_front = non_dominated_sort()
		update_global_repository( pareto_front[0] )
		update_local_repository()
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


