from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np


twoChoice = Model()
# variables are generated in the following order
N = 10
K = 20
twoChoice.add_variable("x_0",1)
for k in range(1,K+1):
    twoChoice.add_variable("x_{}".format(k), 0)
# Adding parameters
twoChoice.add_parameter("rho", .95)
twoChoice.add_parameter("mu", 1)

#setting the system size N
twoChoice.set_system_size("N",N)
# Adding transitions, using a dictionary to represent the update vector
for k in range(1,K+1):
    twoChoice.add_transition({"x_{}".format(k):+1},  "rho*(x_{}**2-x_{}**2)".format(k-1,k))
    if k < K:
        twoChoice.add_transition({"x_{}".format(k):-1},  "mu*(x_{}-x_{})".format(k,k+1))
twoChoice.add_transition({"x_{}".format(K):-1},  "mu*x_{}".format(K))
#adding observables: these are the values tracked by the tool
for k in range(K):
    twoChoice.add_observable("x_{}".format(k), "x_{}".format(k)) 
#finalize initialization
twoChoice.finalize_initialization()


#initialize simulator
simulator = Simulator(twoChoice)
final_time = 2000
points = 100
runs = 100

#integrate mf ode
start = timer()
traj_mf = simulator.MF_simulation(final_time, points)
end = timer()
print("MF simulation time:", (end-start), "seconds")

#integrate corrected mf ode
start = timer()
traj_corr_mf = simulator.correctedMF_simulation(final_time, points)
end = timer()
print("Corrected MF simulation time:", (end-start), "seconds")

#plots the trajectory, plots only some observables
traj_mf.plot(var_to_plot=['x_0','x_1','x_2'])

# Tests (Nicolas) 
print(traj_mf.data[-1,:])
print(traj_corr_mf.data[-1,:])
print(traj_corr_mf.data[-1,:]-traj_mf.data[-1,:])
