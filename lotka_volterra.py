from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np


def lotkavolterra():
    grn = Model()
    # variables are generated in the following order
    grn.add_variable("A", 50)
    grn.add_variable("B", 100)

    # Adding parameters
    grn.add_parameter("k1", 1)
    grn.add_parameter("k2", 0.005)
    grn.add_parameter("k3", 0.6)

    # setting the system size N
    grn.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    grn.add_transition({"A": 1}, "k1*A")
    grn.add_transition({"B": +1, "A": -1}, "k2*A*B")
    grn.add_transition({"B": -1}, "k3*B")

    # adding observables: these are the values tracked by the tool
    grn.add_observable("A", "A")  # syntax: name (to plot) and expression
    grn.add_observable("B", "B")  # syntax: name (to plot) and expression
    grn.finalize_initialization()

    # initialize simulator
    simulator = Simulator(grn)
    final_time = 50
    points = 100
    runs = 500

    print("MF simulation...")
    # integrate mf ode
    start = timer()
    traj_mf = simulator.MF_simulation(final_time, points)
    end = timer()
    print("MF simulation time:", (end - start), "seconds")

    # integrate corrected mf ode
    print("Corrected-MF simulation...")
    start = timer()
    traj_corr_mf = simulator.correctedMF_simulation(final_time, points)
    end = timer()
    print("Corrected MF simulation time:", (end - start), "seconds")

    print("          MF values:", traj_mf.data[points])
    print("Corrected MF values:", traj_corr_mf.data[points])

    # SSA simulation
    print("SSA simulation...")
    start = timer()
    traj_av_ssa = simulator.SSA_simulation(final_time, runs, points, 10)
    end = timer()
    print("SSA simulation time:", (end - start), "seconds")
    print("  Simulation values:", traj_av_ssa.data[points])

    # Prints the errors on the final time point
    print("Error  MF - SSA:", traj_mf.data[points] - traj_av_ssa.data[points])
    print("Error CMF - SSA:", traj_corr_mf.data[points] - traj_av_ssa.data[points])

    return traj_mf, traj_corr_mf, traj_av_ssa
