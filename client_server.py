from correctedMF import *
from time import perf_counter as timer
from time import perf_counter as timer

from correctedMF import *


def client_server_model():
    network = Model()
    # variables are generated in the following order
    network.add_variable("C", 17)
    network.add_variable("Q", 3)

    # Adding parameters
    network.add_parameter("rt", 1.0)
    network.add_parameter("rs", 20.0)

    # setting the system size N
    network.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    network.add_transition({"C": -1, "Q": +1}, "rt*C")
    network.add_transition({"Q": -1, "C": +1}, "rs*Q/(0.01 + Q)")

    # adding observables: these are the values tracked by the tool
    network.add_observable("clients", "C")  # syntax: name (to plot) and expression
    network.add_observable("queue", "Q")  # syntax: name (to plot) and expression
    network.finalize_initialization()

    return network