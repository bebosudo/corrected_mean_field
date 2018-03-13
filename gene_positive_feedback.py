from model import *
from correctedMF import *
from time import perf_counter as timer
import sympy as sp
import numpy as np
from gene_positive_feedback import *


def gene_positive_feedback():
    det = Model()
    # variables are generated in the following order
    det.add_variable("G", 1)
    det.add_variable("Q", 0)
    det.add_variable("GQ", 0)
    det.add_variable("P", 0)
    det.add_variable("D", 0)

    # Adding parameters
    det.add_parameter("kon", 0.001)
    det.add_parameter("koff", 0.0002)
    det.add_parameter("dt", 2.5)
    det.add_parameter("kt", 7)
    det.add_parameter("kdp", 1)
    det.add_parameter("kd", 4)
    det.add_parameter("kmd", 10)
    det.add_parameter("kq", 0.1)
    det.add_parameter("kmq", 1)

    # setting the system size N
    det.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    det.add_transition({"G": -1, "Q": -1, "GQ": +1}, "kon*G*Q")
    det.add_transition({"G": +1, "Q": +1, "GQ": -1}, "koff*GQ")
    det.add_transition({"P": 1}, "dt*G")
    det.add_transition({"P": 1}, "(dt+kt)*GQ")
    det.add_transition({"P": -1}, "kdp*P")
    det.add_transition({"P": -2, "D": +1}, "(kd/2)*P*P")
    det.add_transition({"P": +2, "D": -1}, "kmd*D")
    det.add_transition({"D": -2, "Q": +1}, "(kq/2)*D*D")
    det.add_transition({"D": +2, "Q": -1}, "kmq*Q")

    # adding observables: these are the values tracked by the tool
    det.add_observable("P", "P")  # syntax: name (to plot) and expression
    det.add_observable("G", "G")  # syntax: name (to plot) and expression
    det.add_observable("GQ", "GQ")  # syntax: name (to plot) and expression
    det.add_observable("D", "D")  # syntax: name (to plot) and expression
    det.finalize_initialization()

    return det
