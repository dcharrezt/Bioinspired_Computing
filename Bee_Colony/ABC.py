import numpy as np
import random
import math
import copy

n_iterations = 100
n_solutions = 6
n_dimensions = 2

hive_size = 6
limit = ( hive_size * n_dimensions )/2


min_x = -5
max_x = 5

min_y = -5
max_y = 5

solutions = []
new_solutions = []
best_solution = { "v":[ -1,-1], "func": -1, "fit": -1, 
											"cont": -1 }

def function( x ):
	return x[0]**2 + x[1]**2

def fitness( funct ):
	if( funct >= 0 ):
		return 1 / ( 1 + funct )
	else:
		return 1 + abs( funct )

def init_solutions():
	for i in range( n_solutions ):
		vect = []
		x = random.uniform( min_x, max_x )
		y = random.uniform( min_y, max_y )
		vect.append(x)
		vect.append(y)
		func = function( vect )
		fit = fitness( func )
		solutions.append( { "v":[ x, y], "func": func, "fit": fit, 
											"cont": 0 } )

def get_new_solutions():
	global new_solutions
	new_solutions = []
	for m in range( n_solutions ):
		phi = random.uniform( -1, 1 )
		j = random.randint( 0, n_dimensions-1 )
		k = random.randint( 0, n_solutions-1 )
		for i in range( n_dimensions ):
			if i == j:
				tmp = solutions[m]["v"][i] +  phi * ( solutions[m]["v"][i] -
					  					solutions[k]["v"][i] )
				new_v  = [0]*n_dimensions
				new_v[i] = tmp
				new_v[new_v.index( 0 )] = solutions[m]["v"][new_v.index( 0 )]
				new_func = function( new_v )
				new_fit = fitness( new_func )
				new_solutions.append( { "v": new_v,"func": new_func,"fit": new_fit} )

def roulette_selection():
	rand = random.uniform( 0, 1 )
	sum_fitness = sum([ i["fit"] for i in solutions ])
	s = 0. 
	for i in range( len( solutions )):
		s += (solutions[i]["fit"] / sum_fitness) 
		if s > rand:
			return i

def print_solutions():
	print( " solutions +++++  ")
	print( "V_1\t\t\tV_2\t\t\tFunction\t\tFitness\t\t\tCounter" )
	for i in solutions:
		print( str(i["v"][0])+"\t"+str(i["v"][1])+"\t"+str(i["func"])+"\t"
						+str(i["fit"])+"\t"+str(i["cont"]) )

def print_new_solutions():
	print( "V_1\t\t\tV_2\t\t\tFunction\t\tFitness" )
	for i in new_solutions:
		print( str(i["v"][0])+"\t"+str(i["v"][1])+"\t"+str(i["func"])+"\t"
						+str(i["fit"]))

def compare_solutions():
	for i in range( n_solutions ):
		if new_solutions[i]["fit"] > solutions[i]["fit"]:
			solutions[i]["v"] = copy.deepcopy( new_solutions[i]["v"] )
			solutions[i]["func"] = copy.deepcopy( new_solutions[i]["func"] )
			solutions[i]["fit"] = copy.deepcopy( new_solutions[i]["fit"] )
		else:
			solutions[i]["cont"] += 1

def observer_solution( a ):
	phi = random.uniform( -1, 1 )
	j = random.randint( 0, n_dimensions-1 )
	k = random.randint( 0, n_solutions-1 )
	new_v_j = solutions[a]["v"][j]+phi*(solutions[a]["v"][j]-solutions[k]["v"][j])

	new_v  = [0]*n_dimensions
	new_v[j] =new_v_j
	new_v[new_v.index( 0 )] = solutions[a]["v"][new_v.index( 0 )]
	new_func = function( new_v )
	new_fit = fitness( new_func )

	if new_fit > solutions[a]["fit"]:
		solutions[a]["v"] = copy.deepcopy( new_v )
		solutions[a]["func"] = copy.deepcopy( new_func )
		solutions[a]["fit"] = copy.deepcopy( new_fit )
	else:
		solutions[a]["cont"] += 1

def check_scout_bee():
	best_bee_index = -1
	max_fit = best_solution["fit"]
	for i in range( len( solutions ) ):
		if solutions[i]["cont"] >= limit:
			xs = []
			ys = [] 
			for j in solutions:
				xs.append( j["v"][0] )
				ys.append( j["v"][1] )
			solutions[i]["v"] = [min(xs)+random.uniform(0,1)*(max(xs)-min(xs)) , 
								min(ys)+random.uniform(0,1)*(max(xs)-min(ys))]
			solutions[i]["func"] = function(solutions[i]["v"])
			solutions[i]["fit"] = fitness( solutions[i]["func"])
			solutions[i]["cont"] = 0
		if solutions[i]["fit"] > max_fit:
			max_fit = solutions[i]["fit"]
			best_bee_index = i
	return best_bee_index

def abc():
	global best_solution
	print("Initializing population ------- ")
	init_solutions()
	print_solutions( )
	for i in range( n_iterations ):
		print( "Iteration ++++++ ", i )
		get_new_solutions()
		print_new_solutions( )
		compare_solutions()
		print_solutions()

		for i in range( n_solutions ):
			a = roulette_selection()
			observer_solution( a )
		best_bee_index = check_scout_bee()

		if best_bee_index >= 0:
			best_solution = []
			best_solution = copy.deepcopy( solutions[best_bee_index] )

if __name__=="__main__":
	abc()

	print( "BEST SOLUTION ------- ")
	print( best_solution )