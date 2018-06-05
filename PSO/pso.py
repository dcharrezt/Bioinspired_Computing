import numpy as np
import random

# Parameters 

min_v = -1
max_v = 1

min_x = -5
max_x = 5

min_y = -5
max_y = 5

n_iterations = 3
n_particles = 2

phi_1 = 2.0
phi_2 = 2.0

data = []
best_global = []

def function_fitness( x, y ):
	return x**2 + y**2

def create_particle():
	x = random.uniform(min_x1, max_x1)
	y = random.uniform(min_x2, max_x2)
	v_x = random.uniform(-1, 1)
	v_y = random.uniform(-1, 1)
	p_best = np.inf
	return { "pos": [x, y], "vel": [v_x, v_y], "p_best": [-1,-1,p_best] }

def pso():
	print("Starting PSO")


if __name__=="__main__":
	pso()