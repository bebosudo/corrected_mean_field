from correctedMF import *
from time import perf_counter as timer
import numpy as np
import csv


def gene_mutual_inhibition(ssa_file=None):
    det = Model()
    VOL = 10.0
    # variables are generated in the following order
    det.add_variable("P1", 0)
    det.add_variable("P2", 0)
    det.add_variable("G1", VOL)
    det.add_variable("G2", VOL)
    det.add_variable("M1", 0)
    det.add_variable("M2", 0)
    det.add_variable("M2P1", 0)
    det.add_variable("G1P2", VOL)
    det.add_variable("G2P1", VOL)
    det.add_variable("P1G2", 0)
    det.add_variable("P2G1", 0)

    # Adding parameters
    det.add_parameter("a1", 0.028)
    det.add_parameter("b1", 0.028)
    det.add_parameter("a2", 1.250)
    det.add_parameter("a3", 0.750)
    det.add_parameter("a4", 10.00)
    det.add_parameter("a5", 10.00)
    det.add_parameter("a6", 1.0)
    det.add_parameter("b7", 1.0)
    det.add_parameter("b9", 1.0)

    det.add_parameter("a_r1", 0.005)
    det.add_parameter("b_r1", 0.010)
    det.add_parameter("b2", 1.250)
    det.add_parameter("b3", 0.750)
    det.add_parameter("b4", 10.00)
    det.add_parameter("b5", 10.00)
    det.add_parameter("b6", 1.0)
    det.add_parameter("b8", 57.0)
    det.add_parameter("VOL", VOL)

    # setting the system size N
    det.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    det.add_transition({"P1": -1, "G2": -1, "P1G2": +1}, "(a1/VOL)*P1*G2")
    det.add_transition({"P1": +1, "G2": +1, "P1G2": -1}, "a_r1*P1G2")

    det.add_transition({"P2": -1, "G1": -1, "P2G1": +1}, "(b1/VOL)*P2*G1")
    det.add_transition({"P2": +1, "G1": +1, "P2G1": -1}, "b_r1*P2G1")

    det.add_transition({"M1": +1}, "a2*G1")
    det.add_transition({"M1": +1}, "a3*G1P2")
    det.add_transition({"P1": +1}, "a4*M1")
    det.add_transition({"M1": -1}, "a5*M1")
    det.add_transition({"P1": -1}, "a6*P1")

    det.add_transition({"M2": +1}, "b2*G2")
    det.add_transition({"M2": +1}, "b3*G2P1")
    det.add_transition({"P2": +1}, "b4*M2")
    det.add_transition({"M2": -1}, "b5*M2")
    det.add_transition({"P2": -1}, "b6*P2")

    det.add_transition({"P1": -1, "M2": -1, "M2P1": +1}, "(b8/VOL)*P1*M2")
    det.add_transition({"P1": +1, "M2": +1, "M2P1": -1}, "b7*M2P1")
    det.add_transition({"M2P1": -1}, "b9*M2P1")

    # adding observables: these are the values tracked by the tool
    det.add_observable("P1", "P1")  # syntax: name (to plot) and expression
    det.add_observable("G1", "G1")  # syntax: name (to plot) and expression
    det.add_observable("P2", "P2")  # syntax: name (to plot) and expression
    det.add_observable("G2", "G2")  # syntax: name (to plot) and expression
    det.finalize_initialization()

    return det



