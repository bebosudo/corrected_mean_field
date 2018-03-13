import numpy as np
from model import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy.random as rnd
from math import ceil


# Qui prendiamo un modello con le funzioni numpy create, e costruiamo prima il codice che valuta la funzione
# da integrare e poi la passiamo a odeint.
# poi calcoliamo la correzione.
# qui la funzione prende in input il modello e ritorna media corretta degli osservabili.
# domanda: implemento un gillespie anche per valutare il modello stocastico?


####################################################################################

# class containing a trajectory,
class Trajectory:
    def __init__(self, t, x, desc, labels):
        self.time = t
        self.data = x
        self.labels = labels
        self.description = desc

    def plot(self, var_to_plot=None):
        if var_to_plot is None:
            var_to_plot = self.labels
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(plt.cycler('color', ['r', 'g', 'b', 'c', 'm', 'y', 'k']))
        handles = []
        labels = []
        for v in var_to_plot:
            try:
                i = self.labels.index(v)
                h, = ax.plot(self.time, self.data[:, i])
                handles.append(h)
                labels.append(v)
            except:
                print("Variable", v, "not found")
        fig.legend(handles, labels)
        plt.title(self.description)
        plt.xlabel('Time')
        plt.show()

    def plot_comparing_to(self, trajectory, var_to_plot=None):
        if var_to_plot is None:
            var_to_plot = self.labels
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(plt.cycler('color', ['r', 'r', 'g', 'g', 'b', 'b', 'c', 'c', 'm', 'm', 'y', 'y', 'k', 'k']))
        handles = []
        labels = []
        for v in var_to_plot:
            try:
                i = self.labels.index(v)
                h, = ax.plot(self.time, self.data[:, i])
                handles.append(h)
                labels.append(self.description + " " + v)
                h, = ax.plot(trajectory.time, trajectory.data[:, i], '--')
                handles.append(h)
                labels.append(trajectory.description + " " + v)
            except Exception as e:
                print("Probably variable", v, "not found")
                print("Exception is", e)
        fig.legend(handles, labels)
        plt.title(self.description + " vs " + trajectory.description)
        plt.xlabel('Time')
        plt.show()


#
#
# ####################################################################################
#
#
# # class containing average and variance of trajectories (for stochastic simulation)
# class Trajectory_Statistics:
#     def __init__(self):
#         return


####################################################################################


class Simulator:
    def __init__(self, model):
        self.model = model
        self.t0 = 0
        self.x0 = model.variables.values

    def _unpack(self, x):
        n = self.model.variables.dimension
        phi = x[0:n]  # mean field
        c = x[n:2 * n]  # c term
        d = np.reshape(x[2 * n:], (n, n))  # d term
        return phi, c, d

    def _pack(self, f, dc, dd):
        x = np.concatenate((f.flatten(), dc.flatten(), dd.flatten()))
        return x

    # computes the full vector field for the corrected mean field ODE
    def _corrected_mean_field_ODE(self, x, t):
        phi_t, c_t, d_t = self._unpack(x)
        # compute terms depending on the mean field term
        f, G, J, H = self.model.evaluate_all_vector_fields(phi_t)
        # first term in dc is transposed as tensordot returns a row vector
        dc_dt = np.matmul(c_t, J.transpose()) + np.tensordot(H, d_t, axes=([1, 2], [0, 1]))
        dd_dt = np.matmul(J, d_t) + np.matmul(d_t, J.transpose()) + G / 2
        # repack everything
        dx = self._pack(f, dc_dt, dd_dt)
        return dx

    # computes the vector field for the classic mean field
    def _mean_field_ODE(self, x, t):
        dx = self.model.evaluate_MF_vector_field(x)
        return dx.flatten()

    # computes observables for
    def _compute_corrected_observables(self, x):
        p = np.size(x, 0)
        y = np.zeros((p, self.model.observable_dimension))
        for i in range(p):
            phi, c, d = self._unpack(x[i])
            h, Dh, D2h = self.model.evaluate_all_observables(phi)
            y[i] = h + (np.matmul(c, Dh.transpose()) + np.tensordot(D2h, d,
                                                                    axes=([1, 2], [0, 1]))) / self.model.system_size
        return y

    def _compute_observables(self, x):
        p = np.size(x, 0)
        y = np.zeros((p, self.model.observable_dimension))
        for i in range(p):
            y[i] = self.model.evaluate_observables(x[i])
        return y

    def _generate_time_stamp(self, final_time, points):
        """
        Generates a time stamp from time self.t0 to final_time,
        with points+1 number of points.

        :param final_time: final time of the simulation
        :param points: number of points
        :return: a time stamp numpy array
        """
        step = (final_time - self.t0) / points
        time = np.arange(self.t0, final_time + step, step)
        return time

    def _SSA_single_simulation(self, final_time, time_stamp, model_dimension, trans_number):
        """
        A single SSA simulation run, returns the value of observables

        :param final_time: final simulation time
        :param time_stamp: time array containing time points to save
        :param model_dimension: dimension of the model
        :param trans_number: transitions' number
        :return: the observables computed along the trajectory
        """
        # tracks simulation time and state
        time = 0
        state = self.x0
        # tracks index of the time stamp vector, to save the array
        print_index = 1
        x = np.zeros((len(time_stamp), model_dimension))
        # save initial state
        x[0, :] = self.x0
        # main SSA loop
        trans_code = range(trans_number)
        while time < final_time:
            # compute rates and total rate
            rates = self.model.evaluate_rates(state)
            # sanity check, to avoid negative numbers close to zero
            rates[rates < 1e-14] = 0.0
            total_rate = sum(rates)
            # check if total rate is non zero.
            if total_rate > 1e-14:
                # if so, sample next time and next state and update state and time
                trans_index = rnd.choice(trans_number, p=rates / total_rate)
                delta_time = rnd.exponential(1 / (self.model.system_size * total_rate))
                time += delta_time
                state = state + self.model.transitions[trans_index].update.flatten() / self.model.system_size
            else:
                # If not, stop simulation by skipping to final time
                time = final_time
            # store values in the output array
            while print_index < len(time_stamp) and time_stamp[print_index] <= time:
                x[print_index, :] = state
                print_index += 1
        # computes observables
        y = self._compute_observables(x)
        return y

    def SSA_simulation(self, final_time, runs=100, points=1000, update=1):
        """
        Runs SSA simulation for a given number of runs and returns the average

        :param final_time: final simulation time
        :param runs: number of runs, default is 100
        :param points: number of points to be saved, default is 1000
        :param update: percentage step to update simulation time on screen
        :return: a Trajectory object, containing the average
        """
        time_stamp = self._generate_time_stamp(final_time, points)
        n = self.model.variables.dimension
        m = self.model.transition_number
        average = np.zeros((len(time_stamp), self.model.observable_dimension))
        # LOOP ON RUNS, count from 1
        update_runs = ceil(runs * update / 100.0)
        c = 0
        for i in range(1, runs + 1):
            c = c + 1
            # updates every 1% of simulation time
            if c == update_runs:
                print(ceil(i * 100.0 / runs), "% done")
                c = 0
            y = self._SSA_single_simulation(final_time, time_stamp, n, m)
            # WARNING, works with python 3 only.
            # updating average
            average = (i - 1) / i * average + y / i
        time_stamp = np.reshape(time_stamp, (len(time_stamp), 1))
        trajectory = Trajectory(time_stamp, average, "SSA average", self.model.observable_names)
        return trajectory

    def MF_simulation(self, final_time, points=1000):
        """
        Numerically integrates standard mean field equations

        :param final_time: final simulation time
        :param points: number of points to be saved
        :return:  a trajectory object for model observables
        """
        t = self._generate_time_stamp(final_time, points)
        x = odeint(self._mean_field_ODE, self.x0, t)
        # compute observables
        y = self._compute_observables(x)
        t = np.reshape(t, (len(t), 1))
        trajectory = Trajectory(t, y, "Mean Field", self.model.observable_names)
        return trajectory

    def correctedMF_simulation(self, final_time, points=1000):
        """
        Numerivally integrates the corrected mean field equations

        :param final_time: final simulation time
        :param points: number of points to be saved
        :return:  a trajectory object for corrected model observables
        """
        n = self.model.variables.dimension
        t = self._generate_time_stamp(final_time, points)
        x0 = np.concatenate((self.x0.flatten(), np.zeros(n + n ** 2)))
        x = odeint(self._corrected_mean_field_ODE, x0, t)
        # compute observables
        y = self._compute_corrected_observables(x)
        t = np.reshape(t, (len(t), 1))
        trajectory = Trajectory(t, y, "Corrected Mean Field", self.model.observable_names)
        return trajectory
