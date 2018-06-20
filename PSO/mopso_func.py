



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


def function_1( x, y ):
	return 4*x**2 + 4*x**2

def function_2( x, y):
	return (x-5)**2 + (y-5)**2



def mopso():
	print("****** Starting PSO")
	# create_swarm()
	# print_swarm()
	# evaluating_swarm()

	for i in range( n_iterations ):
		print("Iteration: ", i )


if __name__=="__main__":
	mopso()


