import model
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np


def brusselator():
    det = Model()
    V = 3
    k1 = 1
    k2 = 1
    k3 = 1
    k4 = 1
    detf = 1 / 2
    det.add_variable("A", 2 * V)
    det.add_variable("B", 6 * V)
    det.add_variable("X", 0)
    det.add_variable("Y", 0)
    det.add_variable("D", 0)

    det.add_parameter("V", V)
    det.add_parameter("k1", k1)
    det.add_parameter("k2", k2)
    det.add_parameter("k3", k3)
    det.add_parameter("k4", k4)
    det.add_parameter("det", detf)

    det.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    det.add_transition({"X": 1}, "k1*A")
    det.add_transition({"X": -1, "Y": +1}, "k2/V*B*X")
    det.add_transition({"X": 1, "Y": -1}, "k3/(V*V)*det*X*X*Y")
    det.add_transition({"D": 1, "X": -1}, "k4*X")

    # adding observables: these are the values tracked by the tool
    det.add_observable("X", "X")
    det.add_observable("Y", "Y")
    det.finalize_initialization()

    # initialize simulator
    simulator = Simulator(det)
    final_time = 20
    points = 100

    print("MF simulation...")
    # integrate mf ode
    start = timer()
    traj_mf = simulator.MF_simulation(final_time, points)
    end = timer()
    print("MF simulation time:", (end - start), "seconds")

    traj_mf.plot()

    # integrate corrected mf ode
    print("Corrected-MF simulation...")
    start = timer()
    traj_corr_mf = simulator.correctedMF_simulation(final_time, points)
    end = timer()
    print("Corrected MF simulation time:", (end - start), "seconds")

    print("          MF values:", traj_mf.data[points])
    print("Corrected MF values:", traj_corr_mf.data[points])

    traj_corr_mf.plot()

    ssa = Model()
    ssa.add_variable("A", 2 * V)
    ssa.add_variable("B", 6 * V)
    ssa.add_variable("X", 0)
    ssa.add_variable("Y", 0)
    ssa.add_variable("D", 0)

    ssa.add_parameter("k1", k1)
    ssa.add_parameter("k2", k2)
    ssa.add_parameter("k3", k3)
    ssa.add_parameter("k4", k4)
    ssa.add_parameter("det", detf)
    ssa.add_parameter("V", V)

    ssa.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    ssa.add_transition({"X": 1}, "k1*A")
    ssa.add_transition({"X": -1, "Y": +1}, "k2/V*B*X")
    ssa.add_transition({"X": 1, "Y": -1}, "k3/(V*V)*det*X*(X-1)*Y")
    ssa.add_transition({"D": 1, "X": -1}, "k4*X")

    ssa.add_observable("X", "X")
    ssa.add_observable("Y", "Y")
    ssa.finalize_initialization()

    # initialize simulator
    simulator = Simulator(ssa)
    runs = 1000

    # SSA simulation
    print("SSA simulation...")
    start = timer()
    traj_av_ssa = simulator.SSA_simulation(final_time, runs, points, 5)
    end = timer()
    print("SSA simulation time:", (end - start), "seconds")
    print("  Simulation values:", traj_av_ssa.data[points])

    traj_av_ssa.plot()

    # Prints the errors on the final time point
    print("Error  MF - SSA:", traj_mf.data[points] - traj_av_ssa.data[points])
    print("Error CMF - SSA:", traj_corr_mf.data[points] - traj_av_ssa.data[points])

    return traj_mf, traj_corr_mf, traj_av_ssa
