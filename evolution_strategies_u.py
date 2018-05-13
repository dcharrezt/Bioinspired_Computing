import math


def fitness_function( x_1, x_2):
	return -cos(x_1)*cos(x_2)*math.exp(-(x_1-math.pi)**2 - \
										(x_2-math.pi)**2)

def ep_u_1():
	return True


if __name__ == "__main__":
	ep_u_1()