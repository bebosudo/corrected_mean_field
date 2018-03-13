from toggle_switch import toggle_switch
from gene_positive_feedback  import gene_positive_feedback
import gene_expression
import gene_regulation
from gene_mutual_inhibition import gene_mutual_inhibition
from grima import grima
from hespanha import hespanha
import lotka_volterra
import brusselator
from client_server import client_server_model
from gene_feedback import gene_feedback
from p53 import p53_model
from cardelli import cardelli
from kaic import kaic
from grima2010_heterodimerization import grima2010_heterodimerization
from pathlib import Path
import csv
from correctedMF import *
from time import perf_counter as timer
import numpy as np


import matplotlib.pyplot as plt


def read_csv_output(file_name):
    with open(file_name) as csvfile:
        csvfile = open(file_name)
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        time = [];
        data = [];
        for row in reader:
            time.append(float(row[0]))
            data.append(row[1:])
        csvfile.close()
        matrix_data = np.asanyarray(data)
        return time,matrix_data

# Mean-field, corrected mean-field and SSA are done internally
# The option do_emre does system size expansion using CERENA via Matlab
# If the csv output file is not present then it generates the Matlab script
# If the csv output file is present it will plot the results
def plot_and_save(traj_mf, traj_corr_mf, traj_av_ssa, name, plot_emre):

    if plot_emre is True:
        # we expect a file called name_emre.csv
        filename = "{}_emre.csv".format(name)
        output_file = Path(filename)
        if output_file.exists():
            print("Reading from EMRE file:" + filename)
            emre_time, emre_data = read_csv_output(filename)
        else:
            raise FileExistsError("EMRE output file {} not found.\nGenerate and run CERENA first.".format(filename))

    for v in traj_mf.labels:
        # One plot for each observable
        i = traj_mf.labels.index(v)

        handles = []
        labels = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(plt.cycler('color', ['r', 'g', 'b', 'c', 'm', 'y', 'k']))

        hmf, = ax.plot(traj_mf.time, traj_mf.data[:, i])
        handles.append(hmf)
        labels.append('MF')

        h_corr_mf, = ax.plot(traj_corr_mf.time, traj_corr_mf.data[:, i])
        handles.append(h_corr_mf)
        labels.append('Corr MF')

        h_ssa, = ax.plot(traj_av_ssa.time, traj_av_ssa.data[:, i])
        handles.append(h_ssa)
        labels.append("SSA")

        if plot_emre is True:
            h_emre, = ax.plot(emre_time, emre_data[:, i], linestyle = 'None', marker='.')
            handles.append(h_emre)
            labels.append("EMRE")

        fig.legend(handles, labels)
        plt.title(v)
        plt.xlabel('Time')
        plt.show()

        fig.savefig(name + "_" + v + ".pdf", bbox_inches='tight')

def execute_case_study(model, name, options = {'final_time':2, 'ssa_runs':1000, \
    'do_emre': False, 'matlab_path':'/Users/mirco/Dropbox/MATLAB/CERENA/mean_correction/', \
    'do_ssa': False}):
    # initialize simulator
    simulator = Simulator(model)
    final_time = options['final_time']
    points = 100

    print("MF simulation...")
    # integrate mf ode
    start = timer()
    traj_mf = simulator.MF_simulation(final_time, points)
    end = timer()
    print("MF simulation time:", (end - start), "seconds")

    # integrate corrected mf ode
    print("Corrected-MF simulation...")
    start = timer()
    traj_corr_mf = simulator.correctedMF_simulation(final_time, points)
    end = timer()
    print("Corrected MF simulation time:", (end - start), "seconds")

    print("          MF values:", traj_mf.data[points])
    print("Corrected MF values:", traj_corr_mf.data[points])

    # SSA simulation
    ssa_file = name + "_ssa.csv"

    if options['do_ssa'] is True:
        print("SSA simulation...")
        runs = options['ssa_runs']
        start = timer()
        traj_av_ssa = simulator.SSA_simulation(final_time, runs, points, 10)
        end = timer()
        print("SSA simulation time:", (end - start), "seconds")
        print("  Simulation values:", traj_av_ssa.data[points])
        # Write solution to disk
        output_matrix = np.concatenate((traj_av_ssa.time, traj_av_ssa.data), axis = 1)
        np.savetxt(ssa_file, output_matrix, delimiter=',')
        print("Wrote SSA solution to " + ssa_file)
    else:
        # Read model from file
        try:
            print("Read SSA simulations from file: " + ssa_file)
            time,data = read_csv_output(ssa_file)
            # Inherits labels from other computations
            traj_av_ssa = Trajectory(time, data, "SSA", traj_mf.labels)
        except:
            print("Maybe file SSA output:{} not found".format(ssa_file))
            raise

    # Prints the errors on the final time point
    print("Error  MF - SSA:", traj_mf.data[points] - traj_av_ssa.data[points])
    print("Error CMF - SSA:", traj_corr_mf.data[points] - traj_av_ssa.data[points])

    if options['do_emre'] is True:
        print("Generating EMRE CERENA files to " + options['matlab_path'])
        model.generate_CERENA(name, options['matlab_path'], final_time)

    return traj_mf, traj_corr_mf, traj_av_ssa


# CERENA output is saved here
matlab_path = '/Users/mirco/Dropbox/MATLAB/CERENA/mean_correction/'
plot_emre = True
# Standard options for whole experiment
options = {'do_ssa': True, 'do_emre': True, 'matlab_path': matlab_path}

# model = toggle_switch()
# name = 'toggle_switch'
# options['final_time'] = 100
# options['ssa_runs'] = 1000
# traj_mf, traj_corr_mf, traj_av_ssa = execute_case_study(model, name, options)
# plot_and_save(traj_mf, traj_corr_mf, traj_av_ssa, name, plot_emre=plot_emre)
#
# model = gene_expression()
# name = 'gene_expression'
# options['final_time'] = 80
# options['ssa_runs'] = 2000
# traj_mf, traj_corr_mf, traj_av_ssa = execute_case_study(model, name, options)
# plot_and_save(traj_mf, traj_corr_mf, traj_av_ssa, name, plot_emre=plot_emre)
#
# model = gene_regulation()
# name = 'gene_regulation'
# options['final_time'] = 40
# options['ssa_runs'] = 800
# traj_mf, traj_corr_mf, traj_av_ssa = execute_case_study(model, name, options)
# plot_and_save(traj_mf, traj_corr_mf, traj_av_ssa, name, plot_emre=plot_emre)
#
#
#model = grima()
#name = 'grima'
#options['final_time'] = 20
#options['ssa_runs'] = 5000
#traj_mf, traj_corr_mf, traj_av_ssa = execute_case_study(model, name, options)
#plot_and_save(traj_mf, traj_corr_mf, traj_av_ssa, name, plot_emre=plot_emre)

# model = gene_mutual_inhibition()
# name = 'gene_mutual_inhibition'
# options['final_time'] = 250
# options['ssa_runs'] = 1000
# traj_mf, traj_corr_mf, traj_av_ssa = execute_case_study(model, name, options)
# plot_and_save(traj_mf, traj_corr_mf, traj_av_ssa, name, plot_emre=plot_emre)

# model = gene_feedback()
# name = 'gene_feedback'
# options['final_time'] = 20
# options['ssa_runs'] = 1000
# traj_mf, traj_corr_mf, traj_av_ssa = execute_case_study(model, name, options)
# plot_and_save(traj_mf, traj_corr_mf, traj_av_ssa, name, plot_emre=plot_emre)
#
# model = cardelli()
# name = 'simple_cardelli'
# options['final_time'] = 2
# options['ssa_runs'] = 1000
# traj_mf, traj_corr_mf, traj_av_ssa = execute_case_study(model, name, options)
# plot_and_save(traj_mf,traj_corr_mf,traj_av_ssa, name, plot_emre=plot_emre)
#
# model = grima2010_heterodimerization()
# name = 'grima_heterodimerization'
# options['final_time'] = 0.1
# options['ssa_runs'] = 2000
# traj_mf, traj_corr_mf, traj_av_ssa = execute_case_study(model, name, options)
# plot_and_save(traj_mf,traj_corr_mf,traj_av_ssa, name, plot_emre=plot_emre)
#
# model = kaic()
# name = 'kaic'
# options['final_time'] = 20
# options['ssa_runs'] = 100
# traj_mf, traj_corr_mf, traj_av_ssa = execute_case_study(model, name, options)
# plot_and_save(traj_mf,traj_corr_mf,traj_av_ssa, name, plot_emre=plot_emre)


# In this model EMRE fails because it expects a polynomial expression
# model = client_server_model()
# name = 'client_server'
# options['final_time'] = 5
# options['ssa_runs'] = 5
# traj_mf, traj_corr_mf, traj_av_ssa = execute_case_study(model, name, options)
# plot_and_save(traj_mf, traj_corr_mf, traj_av_ssa, name, plot_emre=False)

# This needs fixing because deterministic and stochastic models are different
#traj_mf,traj_corr_mf,traj_av_ssa = hespanha()
#plot_and_save(traj_mf,traj_corr_mf,traj_av_ssa, 'hespanha')

# This needs fixing because deterministic and stochastic models are different
# traj_mf,traj_corr_mf,traj_av_ssa = brusselator()
# plot_and_save(traj_mf,traj_corr_mf,traj_av_ssa, 'brusselator')

#
# This model issues a segmentation fault
#
# traj_mf,traj_corr_mf,traj_av_ssa = p53_model()
# plot_and_save(traj_mf,traj_corr_mf,traj_av_ssa, 'p53')

#
# Problem with homeoreactions...
#
# traj_mf,traj_corr_mf,traj_av_ssa = genepositivefeedback()
# plot_and_save(traj_mf,traj_corr_mf,traj_av_ssa, 'gene_positive_feedback')

model = gene_positive_feedback()
name = 'gene_positive_feedback'
options['final_time'] = 2000
options['ssa_runs'] = 100
traj_mf, traj_corr_mf, traj_av_ssa = execute_case_study(model, name, options)
plot_and_save(traj_mf, traj_corr_mf, traj_av_ssa, name, plot_emre=plot_emre)
