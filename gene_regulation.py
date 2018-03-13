from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np


def gene_regulation():
    grn = Model()
    # variables are generated in the following order
    grn.add_variable("G", 1)
    grn.add_variable("M", 0)
    grn.add_variable("P", 0)
    grn.add_variable("GP", 0)
    # Adding parameters
    grn.add_parameter("km", 1.0)
    grn.add_parameter("kp", 0.005)
    grn.add_parameter("kb", 0.6)
    grn.add_parameter("ku", 0.001)
    grn.add_parameter("dm", 0.0001)
    grn.add_parameter("dp", 0.005)
    # setting the system size N
    grn.set_system_size("N", 1)
    # Adding transitions, using a dictionary to represent the update vector
    grn.add_transition({"M": 1}, "km*G")
    grn.add_transition({"P": 1}, "kp*M")
    grn.add_transition({"G": -1, "P": -1, "GP": 1}, "kb*G*P")
    grn.add_transition({"GP": -1, "G": 1, "P": 1}, "ku*GP")
    grn.add_transition({"M": -1}, "dm*M")
    grn.add_transition({"P": -1}, "dp*P")

    # adding observables: these are the values tracked by the tool
    grn.add_observable("mRNA", "M")  # syntax: name (to plot) and expression
    grn.add_observable("protein", "P")
    grn.finalize_initialization()
    return grn