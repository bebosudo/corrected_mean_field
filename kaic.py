from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np


def kaic():
    grn = Model()
    # variables are generated in the following order
    grn.add_variable("OO", 400)
    grn.add_variable("OP", 100)
    grn.add_variable("PO", 100)
    grn.add_variable("PP", 200)

    # Adding parameters
    grn.add_parameter("k", 0.01)
    grn.add_parameter("dk", 1.02)
    grn.add_parameter("k2", 1.0)

    # setting the system size N
    grn.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    grn.add_transition({"OO": -1, "OP": 1}, "k*dk*OO*PP")
    grn.add_transition({"OP": -1, "PP": 1}, "k*OP*PP")
    grn.add_transition({"PP": -1, "PO": 1}, "k*dk*OO*PP")
    grn.add_transition({"PO": -1, "OO": 1}, "k*PO*OO")
    grn.add_transition({"PP": -1, "PO": 1}, "k2*PP")
    grn.add_transition({"OO": -1, "OP": 1}, "k2*OO")

    # adding observables: these are the values tracked by the tool
    grn.add_observable("OO", "OO")  # syntax: name (to plot) and expression
    grn.add_observable("OP", "OP")  # syntax: name (to plot) and expression
    grn.finalize_initialization()
    return grn