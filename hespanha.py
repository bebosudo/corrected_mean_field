from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np
from hespanha import *


def hespanha():
    det = Model()
    V = 1
    x10 = 10 * V
    x20 = 25 * V
    i0 = V
    c1 = 52
    c2 = 0.80
    c3 = 0.04
    c4 = 15
    detf = 1 / 2

    det.add_variable("X1", x10)
    det.add_variable("X2", x20)
    det.add_variable("I", i0)

    det.add_parameter("V", V)
    det.add_parameter("c1", c1)
    det.add_parameter("c2", c2)
    det.add_parameter("c3", c3)
    det.add_parameter("c4", c4)
    det.add_parameter("det", detf)

    det.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    det.add_transition({"X1": 40}, "c1*I")
    det.add_transition({"X1": -2}, "(c2*det/V)*X1*X1")
    det.add_transition({"X2": 15}, "(c3*det/V)*X1*X1")
    det.add_transition({"X2": -1}, "c4*X2")

    # adding observables: these are the values tracked by the tool
    det.add_observable("X1", "X1")
    det.add_observable("X2", "X2")
    det.finalize_initialization()

    # initialize simulator
    simulator = Simulator(det)
    final_time = 0.1
    points = 100

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

    ssa = Model()
    V = 1
    # variables are generated in the following order
    ssa.add_variable("X1", x10)
    ssa.add_variable("X2", x20)
    ssa.add_variable("I", i0)

    # Adding parameters
    ssa.add_parameter("V", V)
    ssa.add_parameter("c1", c1)
    ssa.add_parameter("c2", c2)
    ssa.add_parameter("c3", c3)
    ssa.add_parameter("c4", c4)
    ssa.add_parameter("det", detf)

    # setting the system size N
    ssa.set_system_size("N", 1)

    ssa.add_transition({"X1": 40}, "c1*I")
    # change rates for homeoreactions...
    ssa.add_transition({"X1": -2}, "(c2*det/V)*X1*(X1-1)")
    ssa.add_transition({"X2": 15}, "(c3*det/V)*X1*(X1-1)")
    ssa.add_transition({"X2": -1}, "c4*X2")

    # adding observables: these are the values tracked by the tool
    ssa.add_observable("X1", "X1")
    ssa.add_observable("X2", "X2")
    ssa.finalize_initialization()

    # initialize simulator
    simulator = Simulator(ssa)
    runs = 6000

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
