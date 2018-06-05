import numpy as np
import random

# Parameters 

min_v = -1
max_v = 1

min_x = -5
max_x = 5

min_y = -5
max_y = 5

n_iterations = 100
n_particles = 6
n_dimesions = 2

phi_1 = 2.0
phi_2 = 2.0

data = []
best_global = { "x": -1, "y": -1, "fitness": np.inf }

def function_fitness( x, y ):
	return x**2 + y**2

def create_particle():
	x = random.uniform(min_x, max_x)
	y = random.uniform(min_y, max_y)
	v_x = random.uniform(-1, 1)
	v_y = random.uniform(-1, 1)
	fitness = np.inf
	p_best = np.inf
	return { "pos": [x, y], "vel": [v_x, v_y], "fitness": fitness,\
						"p_best": {"x":-1,"y":-1,"fitness": p_best } } 

def create_swarm():
	for i in range( n_particles ):
		data.append( create_particle() )

def print_swarm():
	for part in data:
		print( "x = "+str(part["pos"][0])+
			   " y = "+str(part["pos"][1])+
			   " v_x = "+str(part["vel"][0])+
			   " v_y = "+str(part["vel"][1]) )

def updating_swarm():
	for part in data:
		for i in range( n_dimesions ):
			w = random.random()
			rand_1 = random.random()
			rand_2 = random.random()
			V = w*part["pos"][i] + phi_1*rand_1*(part["p_best"]["x"]-\
				part["pos"][i] ) + phi_2*rand_2*(best_global["x"]-part["pos"][i])
			part["pos"][i] += V

def evaluating_swarm():
	print("****** Fitness")
	for part, i in zip(data, range(n_particles)):
		part["fitness"] = function_fitness( part["pos"][0], part["pos"][1] )
		if part["p_best"]["fitness"] > part["fitness"]:
			part["p_best"]["x"] = part["pos"][0] 
			part["p_best"]["y"] = part["pos"][1] 
			part["p_best"]["fitness"] = part["fitness"]
		if best_global["fitness"] > part["fitness"]:
			best_global["x"] = part["pos"][0]
			best_global["y"] = part["pos"][1]
			best_global["fitness"] = part["fitness"]
		print(str(i)+" ) "+str(part["fitness"]))

def pso():
	print("****** Starting PSO")
	create_swarm()
	print_swarm()
	evaluating_swarm()

	for i in range( n_iterations ):
		print("Iteration: ", i )
		print("Best particle so far: ")
		print("x = "+str(best_global["x"])+" y = "+str(best_global["y"])+\
					" fitness " +str(best_global["fitness"]))
		print("Next swarm ")
		updating_swarm()
		print_swarm()
		evaluating_swarm()


if __name__=="__main__":
	pso()