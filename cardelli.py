from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np


def cardelli():
    model = Model()
    # variables are generated in the following order
    model.add_variable("B1", 1)
    model.add_variable("B2", 1)

    # Adding parameters
    model.add_parameter("k1", 1)
    model.add_parameter("k2", 1)
    model.add_parameter("k3", 1)

    # setting the system size N
    model.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    model.add_transition({"B1": 1}, "k1*B1")
    model.add_transition({"B2": 1}, "k2*B2")
    model.add_transition({"B1": -1, "B2":-1}, "k3*B1*B2")

    # adding observables: these are the values tracked by the tool
    model.add_observable("B1", "B1")  # syntax: name (to plot) and expression
    model.finalize_initialization()

    return model

