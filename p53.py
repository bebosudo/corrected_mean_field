from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np


# This model issues a segmentation fault
def p53_model():
    grn = Model()
    # variables are generated in the following order
    k1 = 90
    k2 = 0.002
    k3 = 1.7
    k4 = 1.1
    k5 = 0.93
    k6 = 0.96
    k7 = 0.01
    grn.add_variable("P53", 70)
    grn.add_variable("Prec", 30)
    grn.add_variable("MDM", 60)

    # Adding parameters
    grn.add_parameter("k1", k1)
    grn.add_parameter("k2", k2)
    grn.add_parameter("k3", k3)
    grn.add_parameter("k4", k4)
    grn.add_parameter("k5", k5)
    grn.add_parameter("k6", k6)
    grn.add_parameter("k7", k7)

    # setting the system size N
    grn.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    grn.add_transition({"P53": 1}, "k1")
    grn.add_transition({"P53": -1}, "k2*P53")
    grn.add_transition({"P53": -1}, "k3*P53*MDM/(k7 + P53)")
    grn.add_transition({"Prec": +1}, "k4*P53")
    grn.add_transition({"Prec": -1, "MDM": +1}, "k5*Prec")
    grn.add_transition({"MDM": -1}, "k6*MDM")

    # adding observables: these are the values tracked by the tool
    grn.add_observable("P53", "P53")  # syntax: name (to plot) and expression
    # grn.add_observable("Prec", "Prec") #syntax: name (to plot) and expression
    # grn.add_observable("MDM", "MDM") #syntax: name (to plot) and expression
    grn.finalize_initialization()

    # initialize simulator
    simulator = Simulator(grn)
    final_time = 120
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
    traj_corr_mf.plot_comparing_to(traj_mf)
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
