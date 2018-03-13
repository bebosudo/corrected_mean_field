from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def toggle_switch():
    network = Model()
    # variables are generated in the following order
    network.add_variable("I", 1)
    network.add_variable("MA", 0)
    network.add_variable("MB", 0)
    network.add_variable("PA", 0)
    network.add_variable("PB", 0)
    network.add_variable("SA", 0)
    network.add_variable("SB", 0)

    # Adding parameters
    network.set_system_size("N", 1)
    # Adding transitions, using a dictionary to represent the update vector
    network.add_transition({"MA": 1}, "0.05*I")
    network.add_transition({"MB": 1}, "0.05*I")
    network.add_transition({"MA": -1}, "0.1*MA")
    network.add_transition({"MB": -1}, "0.1*MB")
    network.add_transition({"PA": -1}, "0.1*PA")
    network.add_transition({"PB": -1}, "0.1*PB")
    network.add_transition({"MA": -1, "SA": +1}, "5*MA")
    network.add_transition({"MB": -1, "SB": +1}, "5*MB")
    network.add_transition({"MB": -1, }, "20*SA*MB")
    network.add_transition({"MA": -1, }, "20*SB*MA")
    network.add_transition({"SA": -1}, "0.01*SA")
    network.add_transition({"SB": -1}, "0.01*SB")
    network.add_transition({"PA": +1}, "10*SA")
    network.add_transition({"PB": +1}, "10*SB")

    # adding observables: these are the values tracked by the tool
    network.add_observable("proteinA", "PA")  # syntax: name (to plot) and expression
    network.add_observable("proteinB", "PB")
    network.finalize_initialization()
    return network