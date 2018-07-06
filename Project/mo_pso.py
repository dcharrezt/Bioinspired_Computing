import random
import copy
import numpy as np
import matplotlib.pyplot as plt

solution_dataset_cost = "datasets/small_cost.txt"
solution_dataset_delay = "datasets/small_delay.txt"

data = []
global_repository = []
cost_matrix = []
delay_matrix = []

n_cabs = 0 
n_passengers = 0
solution_len = 0

n_iterations = 20	
n_particles = 10
n_dimesions = 2
max_cab_capacity = 4

phi_1 = 1.0
phi_2 = 1.0

class Particle:
	def __init__(self, solution):
		self.solution = solution[:]
		self.velocity = []
		self.cost = np.inf
		self.delay = np.inf
		self.local_repository = []

def read_data():
	global n_cabs
	global n_passengers
	global solution_len

	with open(solution_dataset_cost, 'r') as f:
		for line in f:
			cost_matrix.append( list( [float(n) for n in line.split()] ) )

	with open(solution_dataset_delay, 'r') as f:
		for line in f:
			delay_matrix.append( list( [float(n) for n in line.split()] ) )

	n_cabs = len( cost_matrix ) -1
	n_passengers = len( cost_matrix ) -1
	solution_len = n_cabs + n_passengers

def get_passengers_by_cab( solut ):
	pass_by_cab = []
	tmp = []
	for i in range( len(solut.solution) ):
		if solut.solution[i] != -1:
			tmp.append( solut.solution[i] )
		if solut.solution[i] == -1:
			pass_by_cab.append( tmp )
			tmp = []
	pass_by_cab.append(tmp)
	pass_by_cab = [x for x in pass_by_cab if x != []]
	return pass_by_cab

def generate_solution():
	sol = list( range(1, n_passengers+1) )
	cabs = [-1] * n_cabs
	sol += cabs
	random.shuffle( sol )
	a = Particle(sol)
	pc = get_passengers_by_cab( a )
	a.cost = fitness_cost( pc )
	a.delay = fitness_delay( pc )
	return a

def fitness_cost( pc ):
	total_cost = 0
	for i in pc:
		if len(i) == 1:
			total_cost+= cost_matrix[0][i[0]]
		else:
			for j in range(len(i)-1):
				if j == 0:
					total_cost += cost_matrix[0][i[j]]
				total_cost += cost_matrix[i[j]][i[j+1]]
	return total_cost

def fitness_delay( pc ):
	total_delay = 0
	for i in pc:
		if len(i) == 1:
			total_delay+= delay_matrix[0][i[0]]
		else:
			for j in range(len(i)-1):
				if j == 0:
					total_delay += delay_matrix[0][i[j]]
				total_delay += delay_matrix[i[j]][i[j+1]]
	return total_delay

def create_swarm():
	for i in range( n_particles ):
		tmp = generate_solution()
		tmp.local_repository.append( copy.deepcopy(tmp) )
		data.append( copy.deepcopy(tmp) ) 

def evaluate_swarm():
	for i in data:
		pc = get_passengers_by_cab( i )
		i.cost = fitness_cost( pc )
		i.delay = fitness_delay( pc )

def substract_permutations( solution_1, solution_2 ):
	SS = []
	tmp_1 = list( solution_1 )
	tmp_2 = list( solution_2 )
	for i in range( solution_len ):
		if tmp_1[i] != tmp_2[i]:
			index = tmp_2.index( tmp_1[i] )
			SS.append([i , index])
			tmp = tmp_2[i]
			tmp_2[i] = tmp_2[index]
			tmp_2[index] = tmp 
	return SS

def fix_solutions( batch ):
	for sol in batch:
		something_fishy = True
		while( something_fishy ):
			something_fishy = False
			passenger_counter = 0
			for i in range( solution_len ):
				if sol.solution[i] == -1 or i == solution_len-1:
					if passenger_counter <= max_cab_capacity:
						passenger_counter = 0
					else:
						something_fishy = True
						rand = random.randint(2,passenger_counter-1)
						empty_cab_index = -1

						for j in range( solution_len-1 ):
							if( sol.solution[j]==-1 and 
											sol.solution[j+1]==-1 ):
								empty_cab_index = j+1

						first_passenger_index = i-passenger_counter
						if( empty_cab_index < first_passenger_index ):
							first_passenger_index -= 1
						del sol.solution[empty_cab_index]
						sol.solution.insert( first_passenger_index + rand, -1 )
						passenger_counter = 0
						i = solution_len
				else:
					passenger_counter += 1

def dominate( particle_1, particle_2 ):
	if( ( particle_1.cost < particle_2.cost  and \
		  particle_1.delay < particle_2.delay ) or \
		( particle_1.cost <= particle_2.cost and \
		  particle_1.delay < particle_2.delay ) or \
		( particle_1.cost < particle_2.cost and \
		  particle_1.delay <= particle_2.delay ) ):
		return True
	return False

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

def update_global_repository( pareto_front, from_data ):
	global global_repository
	global_repository = []
	for i in range( len(from_data) ):
		if i in pareto_front:
			global_repository.append( copy.deepcopy(from_data[i]) )

def update_local_repository():
	for i in range( len(data) ):
		print( len(data[i].local_repository), end=" " )
		if len( data[i].local_repository ) == 0:
			data[i].local_repository.append( copy.deepcopy( data[i] ) )
		else:
			new_data = data[i].local_repository + [data[i]]
			new_pareto = non_dominated_sort( new_data )
			data[i].local_repository = []
			for j in range( len( new_data) ):
				if j in new_pareto[0]:
					data[i].local_repository.append( copy.deepcopy( new_data[j])) 

def best_local_particle( particle ):
	rand = random.randint(0, len(particle.local_repository) - 1)
	return particle.local_repository[rand]

def best_global_particle():
	rand = random.randint(0, len( global_repository ) -1 )
	return global_repository[ rand ]

def updating_position( particle, SS ):
	for i in SS:
		tmp = particle.solution[i[0]]
		particle.solution[i[0]] = particle.solution[i[1]]
		particle.solution[i[1]] = tmp

def mo_pso():
	read_data()
	create_swarm()
	evaluate_swarm()
	pareto_front = non_dominated_sort( data )
	update_global_repository( pareto_front[0], data )
	update_local_repository()
	fix_solutions( data )
	for i in range( n_iterations ):
		print( "+++++ Iteration ", i )
		for i in range( len(data) ):
			pLocal = best_local_particle( copy.deepcopy(data[i]) )
			pGlocal = copy.deepcopy(best_global_particle())
			a = substract_permutations( data[i].solution, pGlocal.solution )
			b = substract_permutations( data[i].solution, pLocal.solution )
			SS = list(a + b)
			updating_position(data[i], SS)
			fix_solutions( [data[i]] )
		evaluate_swarm()
		new_swarm = data + global_repository
		pareto_front = non_dominated_sort( new_swarm )
		# print("len ", len(new_swarm))
		update_global_repository( pareto_front[0], new_swarm )
		update_local_repository()
	# print(len(data))
	# print(len(global_repository))

if __name__=="__main__":
	mo_pso()

	print("GLObal")
	for m in data:
		print(m.solution)
		print(m.cost)
		print(m.delay)
	print()

	# plt.plot([ i.cost for i in global_repository ], \
	# 			[i.delay for i in global_repository], 'ro')
	# plt.show()