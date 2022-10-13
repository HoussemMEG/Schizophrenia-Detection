import os
from datetime import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plots import Plotter
import mne

import utils
from featuregen import DFG
import preprocessing
import reader


# Read the csv file containing the ERP data and the
df = pd.read_csv(r'C:\Users\meghnouh\PycharmProjects\Schizophrenia Detection\all_chans_ERP.csv', index_col=[0])
print(df.columns, '\n', df.dtypes)

demographic = pd.read_csv("C:/Mon disque D/Gipsa/6- Schizophrenia diagnosis/dataset/dataset 1/demographic.csv")
diagnosis_dict = dict(zip(demographic.subject, demographic[" group"]))  # 1 SZ 0 CTL
channels = list(df.columns[2:-1])

subjects = df['subject'].unique()  # [[random.randint(0, 81)]]
print("Diagnosis dict\n", diagnosis_dict)
print("\n Channels:\n", channels)

# Verbosity
mne.set_log_level(verbose='WARNING')  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
reader.reader_verbose = False
preprocessing.preprocessing_verbose = True

# Stimuli management
stim_types = [['1', '2', '3'],
              ['1'],
              ['2'],
              ['3']
              ][0]
features_container = dict([(stim, {}) for stim in stim_types])

# Parameters
fs = 1024
n_channel = len(channels)


# Class Parameters
my_reader = reader.Reader()

preprocess = preprocessing.PreProcessing(save=False, save_precision='double', overwrite=True,
                                         shift=False, t_shift=0.0,
                                         sampling_freq=None,
                                         filter=[0.5, 35], n_jobs=-1,
                                         rejection_type=None, reject_t_min=-0.1, reject_t_max=0.6,
                                         ref_channel=None,
                                         crop=True, crop_t_min=-0.1, crop_t_max=0.5, include_t_max=True,
                                         baseline=[None, None])

feature_gen = DFG(method='LARS',
                  f_sampling=fs,
                  version=1, omit=None,  # 0
                  normalize=True,
                  model_freq=list(np.linspace(0.1, 40, 40, endpoint=False)),  # upper limit should always be filtering + 5 Hz (due to filter)
                  damping=None,  # (under-damped 0.008 / over-damped 0.09)
                  alpha=2e-3,  # 2e-3
                  merging_weight=0.55,  # 0.50 old value
                  fit_path=True, ols_fit=True,
                  fast=True,
                  selection=np.arange(0.05, 1.05, 0.05),
                  selection_alpha=None,
                  plot=False,
                  show=True, fig_name="fig name", save_fig=False)

# Plotting class of the EEG signal
plotter = Plotter(disable_plot=True,             # if True disable all plots
                  plot_data=True,                # plot all the data (epochs / evoked)
                  plot_psd=False,                # plot power spectral density (epochs)
                  plot_sensors=False,            # sensor location plot (epochs / evoked)
                  plot_image=False, split=True,  # plot epochs image and ERP (epochs)
                  plot_psd_topomap=False,        # plot power spectral density and topomap (epochs)
                  plot_topo_image=False,         # plot epochs image on topomap (epochs)
                  plot_topo_map=False,           # plot ERPs on topomap (evoked)
                  plot_evoked_joint=False,       # plot ERPs data and the topomap on peaks time (evoked)
                  show=True,                     # to whether show the plotted figures or to save them directly
                  save_fig=False,
                  save_path=os.path.join(os.getcwd(), 'figures'))

# Session creation
date = datetime.now().strftime("%Y-%m-%d %H;%M")
session_folder = os.path.join(os.getcwd(), 'features', date)
backup_folder = os.path.join(os.getcwd(), 'features backup', date)

# Main
for i, subj in enumerate(subjects):
    utils.print_c('\nReading file: {:}/{:}'.format(i+1, len(subjects)), bold=True)
    group = diagnosis_dict[subj]
    data_ = np.empty((3, 3072, n_channel))

    # data reading from CSV
    for stim_i, stim in enumerate(df['condition'].unique()):
        filt = (df['subject'] == subj) & (df['condition'] == stim)  # & (t_min * 1000 <= df['time_ms']) & (df['time_ms'] < t_max * 1000)
        data_[stim_i, :, :] = df.loc[filt, channels]  ### this was changed check it
    # data parsing and pre-processing
    data = my_reader.data_to_mne(data_, s_rate=fs, channels=channels, stim_types=stim_types, subj=subj, category=group)
    plotter.plot(data)
    preprocess.set_data(data, epoch_drop_idx=None, epoch_drop_reason='USER', channel_drop=None)
    data = preprocess.process()
    plotter.plot(data)
    data_ = data.get_data()

    # Feature generation
    for j, stim in enumerate(stim_types):
        utils.print_c('\tStim type: <{:}>  {:}/{:}'.format(stim, j+1, len(stim_types)), 'green')
        features, x0 = feature_gen.generate(data_[j].T * 1e6)
        features, x0 = utils.compress(features), utils.ndarray_to_list(x0)
        plt.show()

    #     # Appending the new dynamical features to the dynamical feature container
        temp = {str(subj): {'features': features,
                            'x0': x0,
                            'subject_info': group}}
        features_container[stim].update(temp)

    # here should be the call for each subject of the generator

utils.save_args(features_container, path=session_folder, save_name='generated_features', verbose=True)
utils.save_args(preprocess._saved_args, verbose=True, path=session_folder, save_name='preprocessing_parameters')
utils.save_args({**feature_gen.parameters, **{'channel_picks': channels}, **{'data_case': 'evoked'}},
                path=session_folder, save_name='DFG_parameters', verbose=True)
