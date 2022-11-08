import os
import time
import itertools
from math import comb
import json

import pandas as pd
import numpy as np
from utils import print_c, execution_time, decompress
# from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from scipy.stats import kurtosis, skew
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.svm import SVC

path = r"../features"
param_files = ['DFG_parameters.json', 'preprocessing_parameters.json']

# parameters
use_x0 = False
read_only = True
do_validation = True
do_test = False  # and do_validation #####

select_stim = [None,
               [1],
               [2],
               [3]][2]
select_channel = [None,
                  [['FCz']]][0]

select_feature = [None,
                  [['mean']]][-1]
# ['energy' 'count_non_zero' 'mean' 'max' 'min' 'pk-pk' 'argmin' 'argmax' 'argmax-argmin' 'sum abs' 'var' 'std'
#  'kurtosis' 'skew' 'max abs' 'argmax abs' 'count above val' 'count below val' 'count in range' 'count out range'
#  'count above mean' 'count below mean']

k_feat = 1 if select_feature is None else len(select_feature[0])
k_chan = 1 if select_channel is None else len(select_channel[0])
highlight_above = 0.75

session = ['2022-09-30 00;35',  # 0
           '2022-10-09 23;15',  # 1
           '2022-10-11 21;04',  # 2
           '2022-10-13 13;20',  # 3
           '2022-10-13 21;33',  # 4
           '2022-10-14 14;50',  # 5
           '2022-10-15 12;55',  # 6
           '2022-10-16 01;34',  # 7
           '2022-10-16 23;01',  # 8 version of dataset1 that is focused around 0.15
           '2022-10-16 23;05',  # 9 version of dataset 8 that has more pars
           '2022-10-26 11;52',
           '2022-11-03 15;14',
           '2022-11-04 10;45'
           ][-1]

never_use_SZ = [42, 38, 41, 37, 34, 54, 72, 53, 43, 76, 39, 57, 31, 25, 48]  # category: 1  / 5
never_use_CTL = [65, 1, 14, 17, 23, 21, 12, 16, 22]  # category: 0  / 3
never_use = never_use_SZ + never_use_CTL
never_use = []

print_c('\nSessions: {:}'.format(session.split('\\')[-1]), 'blue', bold=True)


def argmax(lst):
    return lst.index(max(lst))


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
    print(' Parsimony: {:}'.format(np.array(pars)))
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
           'max abs': np.max(np.abs(x), axis=1),
           'argmax abs': np.argmax(np.abs(x), axis=1),
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
        if use_x0:
            data_df = pd.read_csv(os.path.join(os.getcwd(), '../extracted features', session + '-x_0.csv'), index_col=[0])
        else:
            data_df = pd.read_csv(os.path.join(os.getcwd(), '../extracted features', session + '.csv'), index_col=[0])
    except FileNotFoundError:
        print_c('File not found, reading_only has been set to <False>\n', 'red', bold=True)
        read_only = False

if not read_only:
    data_df = read_data(path_session, use_x0=use_x0, param=param)
    file_name = session + '-x_0.csv' if use_x0 else session + '.csv'
    data_df.to_csv(os.path.join(os.getcwd(), '../extracted features', file_name))
    print_c('File saved at {:}'.format(os.path.join(os.getcwd(), '../extracted features', file_name)), bold=True)

data_df.replace(np.nan, 0, inplace=True)
if do_test:
    test_df = data_df.loc[data_df['subject'].isin(never_use)]
data_df.drop(data_df[data_df['subject'].isin(never_use)].index, inplace=True)  # remove test subjects
data_df.reset_index(inplace=True)
category = data_df.groupby(by='subject')['category'].apply('first').to_numpy()
if do_test:
    category_test = test_df.groupby(by='subject')['category'].apply('first').to_numpy()

# Classifier
clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=[0.5, 0.5], n_components=None, store_covariance=False, tol=0.0001, covariance_estimator=None)
# clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001, covariance_estimator=None)
# clf = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5], reg_param=0.0)
# clf = SVC(C=1.0, kernel='linear')

# Classification
max_acc = [0, 0, 0]
timed = 0

grouped = data_df.groupby(by=['stim', 'feature', 'parsimony'])
if do_test:
    grouped_test = test_df.groupby(by=['stim', 'feature', 'parsimony'])

select_stim = data_df['stim'].unique() if select_stim is None else select_stim
for i, stim in enumerate(select_stim):
    j = 0
    select_channel = list(itertools.combinations(channels, k_chan)) if select_channel is None else select_channel
    for channel in select_channel:
        j += 1
        channel = list(channel)
        tic = time.time()
        k = 0

        select_feature = list(itertools.combinations(data_df['feature'].unique(), k_feat)) if select_feature is None else select_feature
        for feature in select_feature:
            k += 1

            score_mem, score_mem_val = [], []

            if do_validation:
                table = [['Parsimony (%)'], ['Train (%)'], ['Validation (%)'], ['Remark']]
            else:
                table = [['Parsimony (%)'], ['Train (%)'], ['Remark']]

            for pars in parsimony:
                feature = list(feature)
                data = select_data(grouped, stim, feature, pars, channel)
                if do_validation:
                    all_idx = range(data.shape[0])
                    train_score_memory, val_score_memory = [], []
                    for subj_idx in all_idx:
                        mask = np.ones(category.shape, bool)
                        mask[subj_idx] = False
                        clf.fit(data[mask, :], category[mask])
                        train_score_memory.append(clf.score(data[mask, :], category[mask]))
                        val_score_memory.append(clf.score(data[[subj_idx], :], category[[subj_idx]]))
                    train_score = sum(train_score_memory) / len(train_score_memory)
                    score_mem.append(train_score * 100)
                    val_score = sum(val_score_memory) / len(val_score_memory)
                    score_mem_val.append(val_score)

                else:
                    clf.fit(data, category)
                    train_score = clf.score(data, category)
                    score_mem.append(train_score * 100)

                if train_score >= highlight_above:
                    table[0].append('\033[92m{:}\033[0m'.format(int(pars * 100)))
                    table[1].append('\033[92m{:.1f}\033[0m'.format(train_score * 100))
                    table[-1].append('\033[92mok\033[0m')
                    if do_validation:
                        table[2].append('\033[92m{:.1f}\033[0m'.format(val_score * 100))
                else:
                    table[0].append('{:}'.format(int(pars * 100)))
                    table[1].append('{:.1f}'.format(train_score * 100))
                    table[-1].append('-')
                    if do_validation:
                        table[2].append('{:.1f}'.format(val_score * 100))

                if train_score >= max_acc[0]:
                    max_acc[0] = train_score
                    max_acc[1] = channel
                    max_acc[2] = stim
            score_mem = np.array(score_mem)

            print_c('Stimuli: {:}     <{:}/{:}>'.format(stim, i + 1, len(select_stim)), 'yellow', bold=True)
            n_iter_chan = len(select_channel) if isinstance(select_channel, list) else comb(len(channels), k_chan)
            print_c('\tChannel: {:}   <{:}/{:}>\t{:.1f}s/it'.format(" / ".join(list(channel)), j, n_iter_chan, timed), 'magenta', bold=True)
            n_iter_feature = len(select_feature) if isinstance(select_feature, list) else comb(len(data_df['feature'].unique()), k_feat)
            print_c('\t\tFeature: {:<25}     <{:}/{:}>'.format(" / ".join(list(feature)), k, n_iter_feature), 'blue', bold=True)
            print(tabulate(table, headers='firstrow', tablefmt="rounded_outline"))

            if do_test:
                # best_pars = parsimony[argmax(score_mem_val)] ####
                best_pars = parsimony[np.argmax(score_mem)]
                data = select_data(grouped, stim, feature, best_pars, channel)
                clf.fit(data, category)
                test_data = select_data(grouped_test, stim, feature, best_pars, channel)
                test_score = clf.score(test_data, category_test)
                print('real label', list(category_test))
                print('predicted ', list(clf.predict(test_data)))

                # # plots
                # SZ = data[category == 1]
                # CTL = data[category == 0]
                # fig, ax = plt.subplots()
                # # ax.scatter(SZ[:, 0], SZ[:, 1], color='red', alpha=0.5)
                # # ax.scatter(CTL[:, 0], CTL[:, 1], color='blue', alpha=0.5)
                #
                # # Decision boudary
                # # x1 = np.array([np.min(data[:, 0]), np.max(data[:, 0], axis=0)])
                # # b, w1, w2 = clf.intercept_[0], clf.coef_[0][0], clf.coef_[0][1]
                # # y1 = -(b + x1 * w1) / w2
                # # ax.plot(x1, y1, c='k', linestyle='--', linewidth=1)
                # ax.set_ylim([-0.002, 0.01])
                # SZ = test_data[category_test == 1]
                # CTL = test_data[category_test == 0]
                # ax.scatter(SZ[:, 0], SZ[:, 1], color='red', alpha=0.5)
                # ax.scatter(CTL[:, 0], CTL[:, 1], color='blue', alpha=0.5)

                print('\t\tAccuracy: {:.1f} % ± {:.2f} %\n\t\tTest accuracy: {:.1f} %'.format(score_mem.mean(),
                                                                                              score_mem.std(),
                                                                                              test_score * 100))
            else:
                print('\t\tAccuracy: {:.1f} % ± {:.2f} %'.format(score_mem.mean(), score_mem.std()))
        timed = time.time() - tic

print('Best accuracy obtained for the session {:} is: {:.1f} % for the stim <{:}> and channel {:} using <{:}>'
      ' features and <{:}> channels'.format(session, max_acc[0] * 100, max_acc[2], max_acc[1], k_feat, k_chan))
plt.show()