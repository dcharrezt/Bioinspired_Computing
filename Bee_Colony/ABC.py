import numpy as np
import random
import math

n_iterations = 5
n_solutions = 3
n_dimensions = 2

hive_size = 6
limit = ( hive_size * n_dimensions )/2


min_x = -5
max_x = 5

min_y = -5
max_y = 5

solutions = []

def function( x, y ):
	return x**2 + y**2

def fitness( funct ):
	if( funct >= 0 ):
		return 1 / ( 1 + funct )
	else:
		return 1 + abs( funct )

def init_solutions():
	for i in range( n_solutions ):
		x = random.uniform( min_x, max_x )
		y = random.uniform( min_y, max_y )
		func = function( x, y)
		fit = fitness( func )
		solutions.append( { "v":[ x, y], "func": func, "fit": fit, "cont": 0 } )

def new_solutions():

	for m in range( n_solutions ):
		phi = random.uniform( -1, 1 )
		j = random.randint( 0, n_dimensions-1 )
		k = random.randint( 0, n_solutions-1 )
		for i in range( n_dimensions ):
			if i == j:
				solutions[m]["v"][i] + phi * ( solutions[m]["v"][i] - 
											solutions[k]["v"][i] )

def abc():

	init_solutions()
	print( solutions )
	new_solutions()
	print( solutions )


	for i in range( n_iterations ):
		print( "Iteration ++++++ ", i ) 

if __name__=="__main__":
	abc()