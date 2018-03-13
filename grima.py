from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np


def grima():
    grn = Model()
    # variables are generated in the following order
    V = 10
    k0 = 200
    k1 = 0.5
    k2 = 0.5
    kin = 0.97 * k2
    grn.add_variable("S", 0)
    grn.add_variable("E", V)
    grn.add_variable("C", 0)
    grn.add_variable("P", 0)
    grn.add_variable("I", V)

    # Adding parameters
    grn.add_parameter("V", V)
    grn.add_parameter("k0", k0)
    grn.add_parameter("k1", k1)
    grn.add_parameter("k2", k2)
    grn.add_parameter("kin", kin)

    # setting the system size N
    grn.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    grn.add_transition({"S": 1}, "kin*I")
    grn.add_transition({"S": -1, "E": -1, "C": 1}, "(k0/V)*S*E")
    grn.add_transition({"S": 1, "E": 1, "C": -1}, "k1*C")
    grn.add_transition({"C": -1, "E": 1, "P": 1}, "k2*C")

    # adding observables: these are the values tracked by the tool
    grn.add_observable("substrate", "S")  # syntax: name (to plot) and expression
    grn.add_observable("enzyme", "E")  # syntax: name (to plot) and expression
    grn.finalize_initialization()

    return grn
