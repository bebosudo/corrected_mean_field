from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np


def grima2010_heterodimerization():
    heterodimerization = Model()
    # variables are generated in the following order
    k1 = 500
    k2 = 40
    k3 = 50
    m = 5 # m = 5 in the original model, larger m gives larger error
    heterodimerization.add_variable("X1", 0)
    heterodimerization.add_variable("X2", 0)
    heterodimerization.add_variable("X3", 0)

    # Adding parameters
    heterodimerization.add_parameter("k1", k1)
    heterodimerization.add_parameter("k2", k2)
    heterodimerization.add_parameter("k3", k3)

    # setting the system size N
    heterodimerization.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    heterodimerization.add_transition({"X1": m}, "k1")
    heterodimerization.add_transition({"X2": m}, "k1")
    heterodimerization.add_transition({"X1": -1, "X2": -1, "X3": 1}, "k2*X1*X2")
    heterodimerization.add_transition({"X1": -1}, "k3")
    heterodimerization.add_transition({"X2": -1}, "k3")
    heterodimerization.add_transition({"X3": -1}, "k3")

    # adding observables: these are the values tracked by the tool
    heterodimerization.add_observable("X1", "X1")  # syntax: name (to plot) and expression
    heterodimerization.finalize_initialization()

    return heterodimerization