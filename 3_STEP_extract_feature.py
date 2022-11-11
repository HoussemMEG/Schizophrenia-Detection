import os
import time
import itertools
from math import comb
import json

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC

from utils import print_c, execution_time, decompress, mean
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from scipy.stats import kurtosis, skew
from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import matplotlib.pyplot as plt


path = r"./features"
param_files = ['DFG_parameters.json', 'preprocessing_parameters.json']

# parameters
read_only = True
do_permutation_test = False
diversify_vote = True  # False for the voting use the same pars chan and feature True to use different parameters to vote
k_feat = 1
k_chan = 1
k_stim = 1
highlight_above = 70
testing_split = ['LOO', 'k-fold'][0]
validation_split = ['LOO', 'k-fold'][0]
k_fold = 5
k_fold_test = 2  # 2, 5, 27  /  37 for bad data removed
shuffle = False

# selection
select_stim = [None, [1], [2], [3]][2]
select_channel = [None, [['C3']]][-1]
select_feature = [None, [['mean']]][-1]
k_feat = k_feat if select_feature is None else len(select_feature[0])
k_chan = k_chan if select_channel is None else len(select_channel[0])

# Classifier
clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=[0.5, 0.5], covariance_estimator=None)
# clf = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5], reg_param=0.0)
# clf = LinearSVC(tol=1e-8, class_weight='balanced', max_iter=1e5)


session = ['2022-10-26 10;46',  # 0
           '2022-10-26 10;52',  # 1
           '2022-10-26 11;27',  # 2, good
           '2022-10-26 11;45',  # 3 same as 2, but with 1% increment
           '2022-10-26 11;52',  # 4, this is best (old 12)
           '2022-10-26 18;48',  # 5, same as 4 with 1% increment
           '2022-10-26 18;02',  # 6, same as 4 with all the channels and stims
           '2022-10-31 11;16',  # 7, same as 4 with a zoom on 75
           '2022-10-31 12;08',  # 8, same as 4 with a zoom on 80
           '2022-11-02 14;40',  # 9 permutation test on this
           '2022-10-23 21;43',  # 10 original with all stims and channels of 4
           '2022-11-03 15;13',  # 11 same as 4 but focus on 80 with 10
           '2022-11-03 15;14',  # 12 same as 4 but focus on 80 with 20
           '2022-09-30 00;35',  # 13
           #####
           '2022-09-30 00;35',  # 14
           '2022-10-09 23;15',  # 15
           '2022-10-11 21;04',  # 16
           '2022-10-13 13;20',  # 17
           '2022-10-13 21;33',  # 18
           '2022-10-14 14;50',  # 19
           '2022-10-15 12;55',  # 20
           '2022-10-16 01;34',  # 21
           '2022-10-17 10;24',  # 22
           '2022-10-18 23;54',  # 23
           '2022-10-19 11;48',  # 24
           '2022-10-19 22;08',  # 25
           '2022-10-23 17;19',  # 26
           '2022-10-23 17;34',  # 27
           '2022-10-23 19;15',  # 28
           '2022-10-23 19;16',  # 29
           '2022-10-23 21;43',  # 30
           '2022-10-23 21;46',  # 31
           ##
           '2022-11-04 10;45',  # 1% increment
           '2022-11-04 10;46',  # 2% increment
           '2022-11-10 23;50',
           '2022-11-11 01;01'
           ][-1]

never_use = []
print('Never use', never_use)
# severe: [5, 17, 23, 30, 33, 57, 78]  /  mild: 31, 46, 66, 72, 79  -  those are only stim 2 outliers


def reset_outer_eval():
    return {'train': np.zeros((81 - len(never_use), len(parsimony))),
            'validation': np.zeros((81 - len(never_use), len(parsimony))),
            'test': [],
            'test_category': np.zeros((81 - len(never_use),)),
            'learning': [],
            'selected validation': []}


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
    pars = float("{:.2f}".format(pars))
    if k_stim > 1:
        data = np.hstack([grouped.get_group((s, feature[0], pars))[channel[0]].to_numpy()[:, np.newaxis]
                          for s in stim])
    else:
        if k_chan > 1:
            if k_feat > 1:
                pass
                data = np.hstack(
                    [grouped.get_group((stim, f, pars))[channel].to_numpy() for f in feature])
            else:  # k_feat == 1
                data = grouped.get_group((stim, feature[0], pars))[channel].to_numpy()

        else:  # k_chan == 1
            if k_feat > 1:
                data = np.array(
                    [grouped.get_group((stim, f, pars))[channel[0]].to_numpy() for f in feature]).T
            else:  # k_feat == 1

                data = grouped.get_group((stim, feature[0], pars))[channel[0]].to_numpy()[:, np.newaxis]
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

                n_freq = param['n_freq']
                n_point = param['n_point']
                n_features = param['n_features']

                if VMS.shape[-1] != len(parsimony):
                    repeat = len(parsimony) - VMS.shape[-1]
                    for i in range(repeat):
                        VMS = np.append(VMS, VMS[:, :, [-1]], axis=2)
                        # VMS[:, :, pars_idx] shape: (n_channel, n_features) per subject

                for pars_idx, pars in enumerate(parsimony):
                    # mask = np.zeros((VMS[:, :, :].shape[1]), dtype=bool)
                    # time selection
                    # mask[n_freq * 0:39*2 * n_freq] = True
                    # frequency selection
                    # for i in range(n_point - 1):
                    #     mask[(n_freq * i + 0):(n_freq * (i+1) - 5)] = True

                    # features_dict = feature_extraction(VMS[:, mask, pars_idx])  # if time or frequency selection
                    features_dict = feature_extraction(VMS[:, :, pars_idx])  # initial
                    for feature_name, features in features_dict.items():
                        subj_dict = {'stim': str(stim),
                                     'subject': int(subj),
                                     'category': category,
                                     'parsimony': float("{:.2f}".format(pars)),
                                     'feature': feature_name}
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
           'count above mean': np.array([np.count_nonzero(row[np.where(row >= np.mean(np.abs(row)))]) for row in x]),
           'count below mean': np.array([np.count_nonzero(row[np.where(row <= np.mean(np.abs(row)))]) for row in x]),
           # 'max abs': np.max(np.abs(x), axis=1),
           # 'argmax abs': np.argmax(np.abs(x), axis=1),
           # 'count above val': np.array([np.count_nonzero(row[np.where(row >= 0.05)]) for row in x]),
           # 'count below val': np.array([np.count_nonzero(row[np.where(row <= -0.05)]) for row in x]),
           # 'count in range': np.array([np.count_nonzero(row[np.where((row <= 0.5) & (row >= -0.5))]) for row in x]),
           # 'count out range': np.array([np.count_nonzero(row[np.where((row >= 0.05) | (row <= -0.05))]) for row in x]),
           }

    for key, value in dic.items():
        if value.ndim > 1:
            raise ValueError("Feature {:} not extracted properly, has the dimension {:}".format(key, value.shape))
        if value.shape != (x.shape[0],):
            raise ValueError("Feature not corresponding to the right dimensions")
    return dic


# read and set parameters
print_c('\nSessions: {:}'.format(session.split('\\')[-1]), 'blue', bold=True)
path_session = os.path.join(path, session)
param = read_param(path_session, param_files)
channels = param['channel_picks']
parsimony = np.array(param['selection'] if param['selection'] is not None else param.get('selection_alpha', None))

if read_only:
    try:
        data_df = pd.read_csv(os.path.join(os.getcwd(), 'extracted features', session + '.csv'), index_col=[0])
    except FileNotFoundError:
        read_only = False
        print_c('File not found, reading_only has been set to <False>\n', 'red', bold=True)
if not read_only:
    data_df = read_data(path_session, use_x0=False, param=param)
    data_df.to_csv(os.path.join(os.getcwd(), 'extracted features', session + '.csv'))
    print_c('File saved at {:}'.format(os.path.join(os.getcwd(), 'extracted features', session + '.csv')), bold=True)


data_df.replace(np.nan, 0, inplace=True)
data_df.drop(data_df[data_df['subject'].isin(never_use)].index, inplace=True)  # remove outlier / test subjects
data_df.reset_index(inplace=True)
category = data_df.groupby(by='subject')['category'].apply('first').to_numpy()

if testing_split == 'LOO':
    f_test = LeaveOneOut()
elif testing_split == 'k-fold':
    if never_use:
        f_test = StratifiedKFold(n_splits=37, shuffle=shuffle)
    else:
        f_test = StratifiedKFold(n_splits=27, shuffle=shuffle)
    if k_fold_test == 2:
        print('k_fold_test 2')
        f_test = StratifiedKFold(n_splits=2, shuffle=shuffle)
    if k_fold_test == 5:
        print('k_fold_test 5')
        f_test = StratifiedKFold(n_splits=5, shuffle=shuffle)


if validation_split == 'LOO':
    f_validation = LeaveOneOut()
elif validation_split == 'k-fold':
    f_validation = StratifiedKFold(n_splits=k_fold, shuffle=shuffle)

test_p_value = []
test_val = []
learning_val = []
validation_val = []
n_repeat = 100000 if (do_permutation_test or shuffle) else 1
for _ in range(n_repeat):
    if do_permutation_test:
        print_c(r'/!\ Permutation test label permutation enabled.', 'red', bold=True)
        temp = category.copy()
        np.random.shuffle(category)
        print('\tShuffle quality: {:.1f} %'.format(np.logical_xor(temp, category).mean() * 100))

    exp_eval = {'validation_acc': np.array([0, 0, 0, 0, 0], dtype=np.float64),
                'channel': [],
                'stim': [],
                'predicted_category': np.zeros((5, 81 - len(never_use))),
                'test': 0}
    timed = 0

    grouped = data_df.groupby(by=['stim', 'feature', 'parsimony'])
    y = category
    # print(y)
    # select_stim = list(itertools.combinations(data_df['stim'].unique(), k_stim)) if k_stim > 1 else select_stim
    select_stim = data_df['stim'].unique() if select_stim is None else select_stim
    select_channel = list(itertools.combinations(channels, k_chan)) if select_channel is None else select_channel
    select_feature = list(itertools.combinations(data_df['feature'].unique(), k_feat)) if select_feature is None else select_feature

    for i, stim in enumerate(select_stim):
        if k_stim > 1:
            stim = list(stim)
        # parsimony = [0.80]
        for j, channel in enumerate(select_channel):
            channel = list(channel)
            for k, feature in enumerate(select_feature):
                feature = list(feature)
                tic = time.time()

                outer_score_mem = reset_outer_eval()
                table = [['Parsimony (%)'], ['Train (%)'], ['Validation (%)'], ['Remark']]
                counter_subj = 0
                for idx_learning, subj_test in f_test.split(np.zeros(81 - len(never_use)), y):
                    counter_subj += 1
                    if testing_split == 'LOO':
                        print(counter_subj)
                    gg = list(f_validation.split(idx_learning, y[idx_learning]))
                    for m, pars in enumerate(parsimony):
                        X = select_data(grouped, stim, feature, pars, channel)
                        # For each test subject out and level of parsimony perform the validation and save it in inner-mem
                        inner_score_mem = {'train': [], 'validation': []}
                        counter = -1
                        for temp_idx_train, temp_subj_validation in gg:
                            counter += 1
                            idx_train = idx_learning[temp_idx_train]
                            subj_validation = idx_learning[temp_subj_validation]
                            clf.fit(X[idx_train, :], y[idx_train])
                            inner_score_mem['train'].append(clf.score(X[idx_train, :], y[idx_train]) * 100 * len(idx_train))
                            inner_score_mem['validation'].append(clf.score(X[subj_validation, :], y[subj_validation]) * 100 * len(subj_validation))
                        outer_score_mem['train'][subj_test, m] = sum(inner_score_mem['train']) / (len(idx_learning) * counter)
                        outer_score_mem['validation'][subj_test, m] = sum(inner_score_mem['validation']) / len(idx_learning)

                    # For the test subject out: select the best performing parsimony level on validation set
                    best_pars_idx = np.argwhere(outer_score_mem['validation'][subj_test[0], :] == np.amax(outer_score_mem['validation'][subj_test[0], :]))[-1][-1]
                    # best_parsimony = parsimony[np.argmax(outer_score_mem['validation'][subj_test[0], :])]  # 0 as they should have all the same accuracy
                    best_parsimony = parsimony[best_pars_idx]  # 0 as they should have all the same accuracy
                    # print(best_parsimony)
                    X = select_data(grouped, stim, feature, best_parsimony, channel)
                    clf.fit(X[idx_learning, :], y[idx_learning])  # use learning set to learn instead of training only
                    outer_score_mem['learning'].append(clf.score(X[idx_learning, :], y[idx_learning]) * 100)  # learning score
                    outer_score_mem['selected validation'].append(np.max(outer_score_mem['validation'][subj_test, :]))
                    outer_score_mem['test'].append(clf.score(X[subj_test, :], y[subj_test]) * 100)  # test score
                    outer_score_mem['test_category'][subj_test] = clf.predict(X[subj_test, :])  # predicted label

                done_once = False
                for m, pars in enumerate(parsimony):
                    train_acc_pars = outer_score_mem['train'][:, m].mean()
                    val_acc_pars = outer_score_mem['validation'][:, m].mean()
                    if outer_score_mem['train'][:, m].mean() >= highlight_above:
                        table[0].append('\033[92m{:}\033[0m'.format(int(pars * 100)))
                        table[1].append('\033[92m{:.1f}\033[0m'.format(train_acc_pars))
                        table[2].append('\033[92m{:.1f}\033[0m'.format(val_acc_pars))
                        table[-1].append('\033[92m OK\033[0m')
                    else:
                        table[0].append('{:}'.format(int(pars * 100)))
                        table[1].append('{:.1f}'.format(train_acc_pars))
                        table[2].append('{:.1f}'.format(val_acc_pars))
                        table[-1].append('-')

                    selection_validation = mean(outer_score_mem['selected validation'])
                    min_val = min(exp_eval['validation_acc'][:]) if not done_once else exp_eval['validation_acc'][i_min]
                    if selection_validation > min_val:
                        i_min = np.argmin(exp_eval['validation_acc']) if not done_once else i_min
                        if diversify_vote:
                            done_once = True
                        exp_eval['validation_acc'][i_min] = selection_validation
                        exp_eval['predicted_category'][i_min] = outer_score_mem['test_category']
                        if selection_validation >= max(exp_eval['validation_acc']):
                            exp_eval['channel'] = channel
                            exp_eval['stim'] = stim
                            exp_eval['test'] = mean(outer_score_mem['test'])

                print_c('Stimuli: {:}     <{:}/{:}>'.format(stim, i+1, len(select_stim)), 'yellow', bold=True)
                print_c('\tChannel: {:}   <{:}/{:}>'.format(" / ".join(list(channel)), j+1, len(select_channel)), 'magenta', bold=True)
                print_c('\t\tFeature: {:<20}     <{:}/{:}>\t\t{:.1f}s/it'.format(" / ".join(list(feature)), k+1, len(select_feature), timed), 'blue', bold=True)
                print(tabulate(table, headers='firstrow', tablefmt="rounded_outline"))

                learning = "\t\t Learning: {:.1f} %".format(mean(outer_score_mem['learning']))
                if mean(outer_score_mem['learning']) > highlight_above:
                    learning = "\t\t\033[92m Learning: {:.1f} %\033[0m".format(mean(outer_score_mem['learning']))
                validation = "\t\t Validation: {:.1f} %".format(mean(outer_score_mem['selected validation']))
                if mean(outer_score_mem['selected validation']) > highlight_above:
                    validation = "\t\t\033[92m Validation: {:.1f} %\033[0m".format(mean(outer_score_mem['selected validation']))
                testing = "\t\t Test: {:.1f} %\033[0m".format(mean(outer_score_mem['test']))
                if mean(outer_score_mem['test']) > highlight_above:
                    testing = "\t\t\033[92m Test: {:.1f} %\033[0m".format(mean(outer_score_mem['test']))
                print('\t\tAccuracy: {:.1f} % ± {:.2f} %'.format(outer_score_mem['train'].mean(), outer_score_mem['train'].std()),
                      learning, validation, testing, '\n')

                timed = time.time() - tic

                # print(outer_score_memory['test_category']) # display vote
    print('\nBest validation accuracy obtained for the session {:} is: {:.1f} % for the stim <{:}> and channel {:} using <{:}>'
          ' features and <{:}> channels\n\t5 best validation accuracies: '.format(session, max(exp_eval['validation_acc']), exp_eval['stim'], exp_eval['channel'], k_feat, k_chan), [float('{:.1f}'.format(val)) for val in exp_eval['validation_acc']])

    # print(list(exp_eval['predicted_category'][0, :]))
    voted = np.sum(exp_eval['predicted_category'], axis=0)
    voted = np.array([1 if x > 2.5 else 0 for x in voted])
    print('\t\tVoted accuracy {:.1f} %'.format(100 - np.logical_xor(voted, category).mean() * 100))
    # test_p_value.append(float('{:.1f}'.format(exp_eval['test'])))
    # print_c('\nCompute p val {:}\n'.format(test_p_value), 'cyan', bold=True)

    # # print just to test the 50 % split
    learning_val.append(float('{:.1f}'.format(mean(outer_score_mem['learning']))))
    validation_val.append(float('{:.1f}'.format(mean(outer_score_mem['selected validation']))))
    test_val.append(float('{:.1f}'.format(exp_eval['test'])))
    print_c('\nLearning   {:}'.format(learning_val), 'cyan', bold=True)
    print_c('Validation {:}'.format(validation_val), 'cyan', bold=True)
    print_c('Testing    {:}\n'.format(test_val), 'cyan', bold=True)

""" features
['energy' 'count_non_zero' 'mean' 'max' 'min' 'pk-pk' 'argmin' 'argmax' 'argmax-argmin' 'sum abs' 'var' 'std'
 'kurtosis' 'skew' 'max abs' 'argmax abs' 'count above val' 'count below val' 'count in range' 'count out range'
 'count above mean' 'count below mean']
 
never_use_SZ = [42, 38, 41, 37, 34, 54, 72, 53, 43, 76, 39, 57, 31, 25, 48]  # category: 1  / 5
never_use_CTL = [65, 1, 14, 17, 23, 21, 12, 16, 22]  # category: 0  / 3
never_use = never_use_SZ + never_use_CTL

# After data visual inspection
# never_use = [5, 17, 23, 30, 33, 57, 78]
# severe: [5, 6, 17, 30, 31, 33, 37, 51, 57, 66, 78]
# mild: [10, 16, 23, 32, 47, 72, 74, 79]
# almost good: [28, 39, 45, 46]

# # plot
# SZ = X[category == 1]
# CTL = X[category == 0]
# plt.scatter(SZ, np.zeros_like(SZ), color='red', alpha=0.5)
# plt.scatter(CTL, np.zeros_like(CTL), color='blue', alpha=0.5)
# plt.show()
"""