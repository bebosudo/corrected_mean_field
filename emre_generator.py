from toggle_switch import toggle_switch
import gene_positive_feedback
from gene_expression import gene_expression
from gene_regulation import gene_regulation
from gene_mutual_inhibition import gene_mutual_inhibition
from grima import grima
from grima2010_heterodimerization import grima2010_heterodimerization
from hespanha import hespanha
import lotka_volterra
import brusselator
from client_server import client_server_model
from gene_feedback import gene_feedback
from p53 import p53_model
from cardelli import cardelli
from kaic import kaic
from correctedMF import *

# CERENA output is saved here
matlab_path = '/Users/mirco/Dropbox/MATLAB/CERENA/mean_correction/'

#model = toggle_switch()
#name = 'toggle_switch'
#model.generate_CERENA(name, matlab_path, 100)

model = gene_expression()
name = 'gene_expression'
model.generate_CERENA(name, matlab_path, 2000)

model = gene_regulation()
name = 'gene_regulation'
model.generate_CERENA(name, matlab_path, 40)

model = grima()
name = 'grima'
model.generate_CERENA(name, matlab_path, 20)

model = gene_mutual_inhibition()
name = 'gene_mutual_inhibition'
model.generate_CERENA(name, matlab_path, 250)

#model = gene_feedback()
#name = 'gene_feedback'
#model.generate_CERENA(name, matlab_path, 20)

model = grima2010_heterodimerization()
name = 'grima_heterodimerization'
model.generate_CERENA(name, matlab_path, 0.1)

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