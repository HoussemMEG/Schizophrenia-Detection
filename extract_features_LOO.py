import os
import time
import itertools
from math import comb
import json

import pandas as pd
import numpy as np
from utils import print_c, execution_time, decompress
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from scipy.stats import kurtosis, skew
from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.svm import SVC

path = r"./features"
param_files = ['DFG_parameters.json', 'preprocessing_parameters.json']

# parameters
read_only = True
k_feat = 1
k_chan = 1
highlight_above = 70

# selection
select_stim = [None, [1], [2], [3]][0]
select_channel = [None, [['FCz']]][0]
select_feature = [None, [['mean', 'var']], [['mean']]][0]

session = ['2022-09-30 00;35',  # 0
           '2022-10-09 23;15',  # 1
           '2022-10-11 21;04',  # 2
           '2022-10-13 13;20',  # 3
           '2022-10-13 21;33',  # 4
           '2022-10-14 14;50',  # 5
           '2022-10-15 12;55',  # 6
           '2022-10-16 01;34',  # 7
           '2022-10-17 10;24',  # 8
           '2022-10-18 23;54',  # 9
           # '2022-10-16 23;01',# 10 version of dataset1 that is focused around 0.15
           # '2022-10-16 23;05',# 11 version of dataset 8 that has more pars
           ][6]
print_c('\nSessions: {:}'.format(session.split('\\')[-1]), 'blue', bold=True)

never_use_SZ = [42, 38, 41, 37, 34, 54, 72, 53, 43, 76, 39, 57, 31, 25, 48]  # category: 1  / 5
never_use_CTL = [65, 1, 14, 17, 23, 21, 12, 16, 22]  # category: 0  / 3
never_use = never_use_SZ + never_use_CTL
never_use = []

k_feat = k_feat if select_feature is None else len(select_feature[0])
k_chan = k_chan if select_channel is None else len(select_channel[0])


def argmax(lst):
    return lst.index(max(lst))

def mean(lst):
    return sum(lst) / len(lst)

def read_param(session, param_files):
    # Reading the JSON files
    param = {}
    for file in os.listdir(session):
        if file in param_files:
            param_path = os.path.join(session, file)
            with open(param_path) as f:
                param.update(json.load(f))

    # Parameters reading
    model_freq = np.array(param['model_freq'])
    n_freq = param['n_freq']
    n_point = param['n_point']
    n_features = param['n_features']
    pars = param['selection'] if param['selection'] is not None else param.get('selection_alpha', None)
    data_case = param.get('data_case', 'evoked')
    alpha = param['alpha']
    version = param.get('version', 0)

    # Channel reading
    channels = param['channel_picks']  # Used channels

    # Printing
    print_c(' Data case: ', highlight=data_case)
    print_c(' Version: ', highlight=str(version))
    print_c(' Alpha: ', highlight=alpha)
    print(' Channels: {:}'.format(param['channel_picks']))
    print(' Model frequencies: {:}'.format(model_freq))
    print(' N_freq = {:}'.format(n_freq))
    print_c(' N_point = ', highlight=n_point)
    print(' Parsimony: {:}\n'.format(np.array(pars)))
    return param


def select_data(grouped, stim, feature, pars, channel):
    if k_chan > 1:
        if k_feat > 1:
            pass
            data = np.hstack(
                [grouped.get_group((stim, f, float("{:.2f}".format(pars))))[channel].to_numpy() for f in
                 feature])
        else:  # k_feat == 1
            data = grouped.get_group((stim, feature[0], float("{:.2f}".format(pars))))[
                channel].to_numpy()

    else:  # k_chan == 1
        if k_feat > 1:
            data = np.array(
                [grouped.get_group((stim, f, float("{:.2f}".format(pars))))[channel[0]].to_numpy() for f
                 in feature]).T
        else:  # k_feat == 1
            data = grouped.get_group((stim, feature[0], float("{:.2f}".format(pars))))[
                       channel[0]].to_numpy()[:, np.newaxis]
    return data


@execution_time
def read_data(session, use_x0, param):
    for file in os.listdir(session):
        if file != "generated_features.json":  # read only json file containing features not parameters
            continue

        file_path = os.path.join(session, file)
        with open(file_path) as f:
            data: dict = json.load(f)

        dic = []
        for stim in data.keys():
            for subj, subj_data in tqdm(data[stim].items(), position=0, leave=True):
                if use_x0:
                    VMS = np.array(subj_data['x0'])  # pre np array shape: List(n_channels)(n_features, n_path)
                else:
                    # subj_feat pre np array shape: List(n_channels)(n_features, n_path)
                    VMS = np.array(decompress(subj_data['features'], n_features=param['n_features']))

                # stim: stim
                # subject ID: subj
                category = subj_data['subject_info']
                channels = param['channel_picks']
                parsimony = np.array(
                    param['selection'] if param['selection'] is not None else param.get('selection_alpha', None))

                for pars_idx, pars in enumerate(parsimony):
                    # VMS[:, :, pars_idx] shape: (n_channel, n_features) per subject
                    features_dict = feature_extraction(VMS[:, :, pars_idx])
                    for feature_name, features in features_dict.items():
                        subj_dict = {'stim': stim,
                                     'subject': int(subj),
                                     'category': category,
                                     'parsimony': float("{:.2f}".format(pars)),
                                     'feature': feature_name}
                        # value should have the shape : (n_channels)
                        for ch_idx, channel in enumerate(channels):
                            subj_dict.update({channel: features[ch_idx]})
                        dic.append(subj_dict)

        # columns = ['stim', 'subject', 'category', 'pasimony', *param['channel_picks']]
        data_df = pd.DataFrame(data=dic, index=None)
        return data_df


def feature_extraction(x):
    dic = {'energy': np.sum(x ** 2, axis=1),
           'count_non_zero': np.count_nonzero(x, axis=1),
           'mean': np.mean(x, axis=1),
           'max': np.max(x, axis=1),
           'min': np.min(x, axis=1),
           'pk-pk': np.max(x, axis=1) - np.min(x, axis=1),
           'argmin': np.argmin(x, axis=1),
           'argmax': np.argmax(x, axis=1),
           'argmax-argmin': np.argmax(x, axis=1) - np.argmin(x, axis=1),
           'sum abs': np.sum(np.abs(x), axis=1),
           'var': np.var(x, axis=1),
           'std': np.std(x, axis=1),
           'kurtosis': kurtosis(x, axis=1),
           'skew': skew(x, axis=1),
           # 'max abs': np.max(np.abs(x), axis=1),
           # 'argmax abs': np.argmax(np.abs(x), axis=1),
           # 'count above val': np.array([np.count_nonzero(row[np.where(row >= 0.05)]) for row in x]),
           # 'count below val': np.array([np.count_nonzero(row[np.where(row <= -0.05)]) for row in x]),
           # 'count in range': np.array([np.count_nonzero(row[np.where((row <= 0.5) & (row >= -0.5))]) for row in x]),
           # 'count out range': np.array([np.count_nonzero(row[np.where((row >= 0.05) | (row <= -0.05))]) for row in x]),
           'count above mean': np.array([np.count_nonzero(row[np.where(row >= np.mean(np.abs(row)))]) for row in x]),
           'count below mean': np.array([np.count_nonzero(row[np.where(row <= np.mean(np.abs(row)))]) for row in x]),
           }

    for key, value in dic.items():
        if value.ndim > 1:
            raise ValueError("Feature {:} not extracted properly, has the dimension {:}".format(key, value.shape))
        if value.shape != (x.shape[0],):
            raise ValueError("Feature not corresponding to the right dimensions")
    return dic


# read and set parameters
path_session = os.path.join(path, session)
param = read_param(path_session, param_files)
channels = param['channel_picks']
parsimony = np.array(param['selection'] if param['selection'] is not None else param.get('selection_alpha', None))

if read_only:
    try:
        data_df = pd.read_csv(os.path.join(os.getcwd(), 'extracted features', session + '.csv'), index_col=[0])
    except FileNotFoundError:
        print_c('File not found, reading_only has been set to <False>\n', 'red', bold=True)
        read_only = False

if not read_only:
    data_df = read_data(path_session, use_x0=False, param=param)
    data_df.to_csv(os.path.join(os.getcwd(), 'extracted features', session + '.csv'))
    print_c('File saved at {:}'.format(os.path.join(os.getcwd(), 'extracted features', session + '.csv')), bold=True)


data_df.replace(np.nan, 0, inplace=True)
data_df.drop(data_df[data_df['subject'].isin(never_use)].index, inplace=True)  # remove test subjects
data_df.reset_index(inplace=True)
category = data_df.groupby(by='subject')['category'].apply('first').to_numpy()


# Classifier
clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=[0.5, 0.5], n_components=None, store_covariance=False, tol=0.0001, covariance_estimator=None)
# clf = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5], reg_param=0.0)
# clf = SVC(C=1.0, kernel='linear')

# Classification
max_acc = {'validation_acc': np.array([0, 0, 0, 0, 0], dtype=np.float64),
           'channel': [],
           'stim': []}
timed = 0
skf_test = StratifiedKFold(n_splits=27)
skf_validation = StratifiedKFold(n_splits=5)
grouped = data_df.groupby(by=['stim', 'feature', 'parsimony'])
y = category

select_stim = data_df['stim'].unique() if select_stim is None else select_stim
select_channel = list(itertools.combinations(channels, k_chan)) if select_channel is None else select_channel
select_feature = list(itertools.combinations(data_df['feature'].unique(), k_feat)) if select_feature is None else select_feature

for i, stim in enumerate(select_stim):
    for j, channel in enumerate(select_channel):
        channel = list(channel)

        for k, feature in enumerate(select_feature):
            feature = list(feature)
            tic = time.time()
            outer_score_memory = {'train': np.zeros((81, len(parsimony))),
                                  'validation': np.zeros((81, len(parsimony))),
                                  'test': []}
            table = [['Parsimony (%)'], ['Train (%)'], ['Validation (%)'], ['Remark']]
            idx = list(range(len(y)))

            ## LOO
            for idx_learning, subj_test in skf_test.split(np.zeros(81), y):
            ## k-fold
            # for subj_test in idx:
            #     idx_learning = idx[:]
            #     idx_learning.remove(subj_test)  # 80 subjects
                for m, pars in enumerate(parsimony):
                    X = select_data(grouped, stim, feature, pars, channel)
                    inner_score_memory = {'train': [],
                                          'validation': []}
                    ## LOO
                    # for subj_validation in idx_learning:
                    #     idx_train = idx_learning[:]
                    #     idx_train.remove(subj_validation)  # 79 subject
                    ## k-fold
                    for idx_train, subj_validation in skf_validation.split(X[idx_learning, :], y[idx_learning]):
                        clf.fit(X[idx_train, :], y[idx_train])
                        inner_score_memory['train'].append(clf.score(X[idx_train, :], y[idx_train]) * 100)
                        ## LOO
                        # inner_score_memory['validation'].append(clf.score(X[[subj_validation], :], y[[subj_validation]]) * 100)
                        ## k-fold
                        inner_score_memory['validation'].append(clf.score(X[subj_validation, :], y[subj_validation]) * 100)

                    temp_train = inner_score_memory['train']
                    temp_validation = inner_score_memory['validation']
                    outer_score_memory['train'][subj_test, m] = mean(temp_train)
                    outer_score_memory['validation'][subj_test, m] = mean(temp_validation)

                best_parsimony = parsimony[np.argmax(outer_score_memory['validation'][subj_test, :])]
                X = select_data(grouped, stim, feature, best_parsimony, channel)
                clf.fit(X[idx_learning, :], y[idx_learning])
                # LOO
                # outer_score_memory['test'].append(clf.score(X[[subj_test], :], y[[subj_test]]) * 100)
                # k-fold
                outer_score_memory['test'].append(clf.score(X[subj_test, :], y[subj_test]) * 100)

            for m, pars in enumerate(parsimony):
                train_acc_pars = outer_score_memory['train'][:, m].mean()
                val_acc_pars = outer_score_memory['validation'][:, m].mean()
                if outer_score_memory['train'][:, m].mean() >= highlight_above:
                    table[0].append('\033[92m{:}\033[0m'.format(int(pars * 100)))
                    table[1].append('\033[92m{:.1f}\033[0m'.format(train_acc_pars))
                    table[2].append('\033[92m{:.1f}\033[0m'.format(val_acc_pars))
                    table[-1].append('\033[92m OK\033[0m')
                else:
                    table[0].append('{:}'.format(int(pars * 100)))
                    table[1].append('{:.1f}'.format(train_acc_pars))
                    table[2].append('{:.1f}'.format(val_acc_pars))
                    table[-1].append('-')

                if val_acc_pars > min(max_acc['validation_acc']):
                    i_min = np.argmin(max_acc['validation_acc'])
                    max_acc['validation_acc'][i_min] = val_acc_pars
                    if val_acc_pars >= max(max_acc['validation_acc']):
                        max_acc['channel'] = channel
                        max_acc['stim'] = stim

            print_c('Stimuli: {:}     <{:}/{:}>'.format(stim, i+1, len(select_stim)), 'yellow', bold=True)
            print_c('\tChannel: {:}   <{:}/{:}>'.format(" / ".join(list(channel)), j+1, len(select_channel)), 'magenta', bold=True)
            print_c('\t\tFeature: {:<20}     <{:}/{:}>\t\t{:.1f}s/it'.format(" / ".join(list(feature)), k+1, len(select_feature), timed), 'blue', bold=True)
            print(tabulate(table, headers='firstrow', tablefmt="rounded_outline"))

            test_accuracy = mean(outer_score_memory['test'])
            if test_accuracy > highlight_above:
                print('\t\tAccuracy: {:.1f} % ± {:.2f} % \t\t \033[92m Test accuracy: {:.1f} %\033[0m'
                      .format(outer_score_memory['train'].mean(), outer_score_memory['train'].std(), test_accuracy))
            else:
                print('\t\tAccuracy: {:.1f} % ± {:.2f} % \t\t Test accuracy: {:.1f} %'
                      .format(outer_score_memory['train'].mean(), outer_score_memory['train'].std(), test_accuracy))
            timed = time.time() - tic

print('\nBest validation accuracy obtained for the session {:} is: {:.1f} % for the stim <{:}> and channel {:} using <{:}>'
      ' features and <{:}> channels\nOther max accuracies'.format(session, max(max_acc['validation_acc']), max_acc['stim'], max_acc['channel'], k_feat, k_chan), [float('{:.1f}'.format(val)) for val in max_acc['validation_acc']])


""" features
# ['energy' 'count_non_zero' 'mean' 'max' 'min' 'pk-pk' 'argmin' 'argmax' 'argmax-argmin' 'sum abs' 'var' 'std'
#  'kurtosis' 'skew' 'max abs' 'argmax abs' 'count above val' 'count below val' 'count in range' 'count out range'
#  'count above mean' 'count below mean']
"""