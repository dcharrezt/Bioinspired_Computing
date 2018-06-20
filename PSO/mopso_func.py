import numpy as np
import random



# parameters

n_iterations = 2
n_particles = 6
n_dimesions = 2

min_v = -1
max_v = 1

min_x = 0
max_x = 5

min_y = 0
max_y = 3

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
	print("****** Fitness")
	for part, i in zip(data, range(n_particles)):
		part["fitness_1"] = function_1( part["pos"][0], part["pos"][1] )
		part["fitness_2"] = function_2( part["pos"][0], part["pos"][1] )



		# if part["p_best"]["fitness"] > part["fitness"]:
		# 	part["p_best"]["x"] = part["pos"][0] 
		# 	part["p_best"]["y"] = part["pos"][1] 
		# 	part["p_best"]["fitness"] = part["fitness"]
		# if best_global["fitness"] > part["fitness"]:
		# 	best_global["x"] = part["pos"][0]
		# 	best_global["y"] = part["pos"][1]
		# 	best_global["fitness"] = part["fitness"]
		print(str(i)+" ) "+str(part["fitness_1"])+"\t"+str(part["fitness_2"]))

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
	for i in pareto_front:
		global_repository.append( data[i] )

def update_local_repository():
	for i in range( n_particles ):
		data[i]["repo"].append( data[i] )

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


if __name__=="__main__":
	mopso()


