from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np


def gene_expression():
    grn = Model()
    # variables are generated in the following order
    grn.add_variable("Doff", 1)
    grn.add_variable("Don", 0)
    grn.add_variable("R", 0)
    grn.add_variable("P", 0)

    # Adding parameters
    grn.add_parameter("ton", 0.05)
    grn.add_parameter("toff", 0.05)
    grn.add_parameter("kr", 10)
    grn.add_parameter("gammar", 1)
    grn.add_parameter("kp", 4)
    grn.add_parameter("gammap", 1)
    grn.add_parameter("tonp", 0.0015)

    # setting the system size N
    grn.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    grn.add_transition({"Doff": -1, "Don": +1}, "ton*Doff")
    grn.add_transition({"Doff": +1, "Don": -1}, "toff*Don")
    grn.add_transition({"R": +1}, "kr*Don")
    grn.add_transition({"R": -1}, "gammar*R")
    grn.add_transition({"P": +1}, "kp*R")
    grn.add_transition({"P": -1}, "gammap*P")
    grn.add_transition({"Doff": -1, "Don": +1}, "tonp*P*Doff")

    # adding observables: these are the values tracked by the tool
    grn.add_observable("protein", "P")  # syntax: name (to plot) and expression
    grn.finalize_initialization()

    return grn
