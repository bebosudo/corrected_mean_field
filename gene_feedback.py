from gene_positive_feedback import *


def gene_feedback():
    det = Model()
    # variables are generated in the following order
    det.add_variable("Du", 1)
    det.add_variable("Db", 0)
    det.add_variable("P", 0)

    # Adding parameters
    det.add_parameter("ru", 1.0)
    det.add_parameter("rb", 0.5)
    det.add_parameter("kf", 0.1)
    det.add_parameter("kb", 1.0)
    det.add_parameter("sb", 10.0)
    det.add_parameter("su", 0.5)

    # setting the system size N
    det.set_system_size("N", 1)

    # Adding transitions, using a dictionary to represent the update vector
    det.add_transition({"P": +1}, "ru*Du")
    det.add_transition({"P": +1}, "rb*Db")
    det.add_transition({"Db": -1, "Du": +1}, "kb*Db")
    det.add_transition({"P": -1, "Du": -1, "Db": +1}, "sb*Du*P")
    det.add_transition({"P": +1, "Du": +1, "Db": -1}, "su*Db")
    det.add_transition({"P": -1}, "kf*P")

    # adding observables: these are the values tracked by the tool
    det.add_observable("P", "P")  # syntax: name (to plot) and expression
    det.add_observable("Db", "Db")  # syntax: name (to plot) and expression
    det.finalize_initialization()
    return det
