import numpy as np
import random

x_lower_limit = 0.
x_upper_limit = 5.

y_lower_limit = 0.
y_upper_limit = 3.

data = []

population_size = 5
n_adversaries = 3
beta = 0.5
alpha = 1.

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



def function_1( x, y):
	return 4*(x**2) + 4*(y**2)

def function_2( x, y):
	return (x-5)**2 + (y-5)**2

def generate_individual():
	return {"x": random.uniform(x_lower_limit, x_upper_limit), \
			"y": random.uniform(y_lower_limit, y_upper_limit), \
			"fitness_1": np.inf, \
			"fitness_2": np.inf }

def generate_population():
	for i in range( population_size ):
		data.append( generate_individual() )

def evaluate_population():
	for i in data:
		data["fitness_1"] = function_1( data["x"], data["y"] )
		data["fitness_2"] = function_2( data["x"], data["y"] )

def tournament_selection( n_adversaries ):
	adversaries = np.random.permutation( list( range( population_size ) ) )
	tmp = [ data[i] for i in adversaries[:n_adversaries]]
	return min(tmp, key=lambda item: item["fitness"])

def BLX_crossover( parent_1, parent_2 ):
	m_beta = random.uniform( beta - alpha, beta + alpha )
	
	m_x = parent_1["x"] + m_beta*( parent_2["x"] - parent_1["x"] )
	m_y = parent_1["y"] + m_beta*( parent_2["y"] - parent_1["y"] )
	m_1 = function_1( m_x, m_y )
	m_2 = function_2( m_x, m_y )

	return {"x": m_x, "y": m_y, "fitness_1": m_1, "fitness_2": m_2 }



def minimize_F():
	generate_individual()

if __name__ == "__main__":
	BLX_crossover()
	# print(generate_individual())
	# print( np.array(city_distances).shape )
	# print( np.array(cost_between_cities).shape )
	print("hELLO wORLD")