from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np


sir = Model()
# variables are generated in the following order
N = 10
sir.add_variable("S", 0.2)
sir.add_variable("I", 0.3)
sir.add_variable("R", 0.5)
# Adding parameters
sir.add_parameter("ki", 2)
sir.add_parameter("kr", 1)
sir.add_parameter("ke", 1)
sir.add_parameter("ks", 3)
#setting the system size N
sir.set_system_size("N",N)
# Adding transitions, using a dictionary to represent the update vector
sir.add_transition({"S":-1, "I":1},  "ke*S + ki*I*S")
sir.add_transition({"I":-1, "R":1},  "kr*I")
sir.add_transition({"R":-1, "S":1},  "ks*R")
#adding observables: these are the values tracked by the tool
sir.add_observable("susceptible", "S") 
sir.add_observable("infected", "I")
sir.add_observable("recovered", "R")
#finalize initialization
sir.finalize_initialization()


#initialize simulator
simulator = Simulator(sir)
final_time = 20
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
#traj_mf.plot(var_to_plot=['infected'])

# Tests (Nicolas) 
print(traj_mf.data[-1,:])
print(traj_corr_mf.data[-1,:])
print(traj_corr_mf.data[-1,:]-traj_mf.data[-1,:])

# #SSA simulation
# start = timer()
# traj_av_ssa = simulator.SSA_simulation(final_time, runs, points)
# end = timer()
# print("SSA simulation time:", (end-start), "seconds")


# #compares two trajectories
# traj_corr_mf.plot_comparing_to(traj_av_ssa)






