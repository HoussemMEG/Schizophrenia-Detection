import os
import time
import itertools
from math import comb
import json

import pandas as pd
import numpy as np
from utils import print_c, execution_time, decompress
from tqdm.notebook import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from scipy.stats import kurtosis, skew

# parameters
use_x0 = False
read_only = True
k_feat = 2
k_chan = 1
highlight_above = 0.70


path = r"C:\Users\meghnouh\PycharmProjects\Schizophrenia Detection\features"
param_files = ['DFG_parameters.json', 'preprocessing_parameters.json']
sessions = [os.path.join(path, sess) for sess in os.listdir(path) if sess != "don't read"]

never_use_SZ = [42, 38, 41, 37, 34, 54, 72, 53, 43, 76, 39, 57, 31, 25, 48]  # category: 1
never_use_CTL = [65, 1, 14, 17, 23, 21, 12, 16, 22]                          # category: 0
never_use = never_use_SZ + never_use_CTL
never_use = []


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


@execution_time
def read_data(session, use_x0, param):
    for file in os.listdir(session):
        if file != "generated_features.json":  # read only json file containing features not parameters
            continue

        file_path = os.path.join(session, file)
        with open(file_path) as f:
            data: dict = json.load(f)

        dic = []
        for stim in tqdm(data.keys(), position=0):
            for subj, subj_data in tqdm(data[stim].items(), position=1):
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
                                     'subject': subj,
                                     'category': category,
                                     'parsimony': "{:.2f}".format(pars),
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
           'count above val': np.array([np.count_nonzero(row[np.where(row >= 0.05)]) for row in x]),
           'count below val': np.array([np.count_nonzero(row[np.where(row <= -0.05)]) for row in x]),
           'count in range': np.array([np.count_nonzero(row[np.where((row <= 0.5) & (row >= -0.5))]) for row in x]),
           'count out range': np.array([np.count_nonzero(row[np.where((row >= 0.05) | (row <= -0.05))]) for row in x]),
           'count above mean': np.array([np.count_nonzero(row[np.where(row >= np.mean(np.abs(row)))]) for row in x]),
           'count below mean': np.array([np.count_nonzero(row[np.where(row <= np.mean(np.abs(row)))]) for row in x]),
           }

    for key, value in dic.items():
        if value.ndim > 1:
            raise ValueError("Feature {:} not extracted properly, has the dimension {:}".format(key, value.shape))
        if value.shape != (x.shape[0],):
            raise ValueError("Feature not corresponding to the right dimensions")
    return dic


for sess in sessions:
    if sess.split('\\')[-1] == r"don't read":
        continue
    print_c('\nSessions: {:}'.format(sess.split('\\')[-1]), 'blue', bold=True)

    # read and set parameters
    param = read_param(sess, param_files)
    channels = param['channel_picks']
    parsimony = np.array(param['selection'] if param['selection'] is not None else param.get('selection_alpha', None))

    if not read_only:
        data_df = read_data(sess, use_x0=use_x0, param=param)
        file_name = os.path.basename(sess) + '-x_0.csv' if use_x0 else os.path.basename(sess) + '.csv'
        data_df.to_csv(os.path.join(r'C:\Users\meghnouh\PycharmProjects\Schizophrenia Detection\extracted features', file_name))

    if read_only:
        if use_x0:
            data_df = pd.read_csv(
                os.path.join(r'C:\Users\meghnouh\PycharmProjects\Schizophrenia Detection\extracted features', os.path.basename(sess) + '-x_0.csv'), index_col=[0])
        else:
            data_df = pd.read_csv(
                os.path.join(r'C:\Users\meghnouh\PycharmProjects\Schizophrenia Detection\extracted features', os.path.basename(sess) + '.csv'), index_col=[0])

    data_df.replace(np.nan, 0, inplace=True)
    data_df.drop(data_df[data_df['subject'].isin(never_use)].index, inplace=True)  # remove test subjects
    data_df.reset_index(inplace=True)
    category = data_df.groupby(by='subject')['category'].apply('first').to_numpy()
    print(data_df)

    clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=[0.5, 0.5], n_components=None,
                                     store_covariance=False, tol=0.0001, covariance_estimator=None)
    # clf = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0)

    max_acc = 0
    tic = time.time()
    grouped = data_df.groupby(by=['stim', 'feature', 'parsimony'])
    for i, stim in enumerate(data_df['stim'].unique()):
        print_c('\n\nStimuli: {:}     <{:}/{:}>'.format(stim, i+1, len(data_df['stim'].unique())), 'yellow', bold=True)

        j = 0
        for channel in itertools.combinations(channels, k_chan):
            j += 1
            print_c('\tChannel: {:}   <{:}/{:}>'.format(" / ".join(list(channel)), j, comb(len(channels), k_chan)), 'magenta', bold=True)
            channel = list(channel)

            k = 0
            for feature in itertools.combinations(data_df['feature'].unique(), k_feat):
                k += 1
                print_c('\t\tFeature: {:}     <{:}/{:}>'.format(" / ".join(list(feature)), k, comb(len(data_df['feature'].unique()), k_chan)), 'blue', bold=True)
                score_mem = []
                for pars in parsimony:
                    feature = list(feature)

                    if k_chan > 1:
                        if k_feat > 1:
                            pass
                            data = np.hstack([grouped.get_group((stim, f, float("{:.2f}".format(pars))))[channel].to_numpy() for f in feature])
                        else:  # k_feat == 1
                            data = grouped.get_group((stim, feature[0], float("{:.2f}".format(pars))))[channel].to_numpy()

                    else:  # k_chan == 1
                        if k_feat > 1:
                            data = np.array([grouped.get_group((stim, f, float("{:.2f}".format(pars))))[channel[0]].to_numpy() for f in feature]).T
                        else:  # k_feat == 1
                            data = grouped.get_group((stim, feature[0], float("{:.2f}".format(pars))))[channel[0]].to_numpy()[:, np.newaxis]

                    clf.fit(data, category)
                    score = clf.score(data, category)
                    score_mem.append(score)
                    if score >= highlight_above:
                        print_c('\t\t\t\tParsimony level: {:.2f} % \t Accuracy: {:.2f} %   good'.format(pars, 100 * score), 'green', bold=True)
                    else:
                        print('\t\t\t\tParsimony level: {:.2f} % \t Accuracy: {:.2f} %'.format(pars, 100 * score))
                    if score >= max_acc:
                        max_acc = score
                score_mem = np.array(score_mem) * 100
                print('\t\t\t\t\t\t\t\t\t\t\t Accuracy: {:.2f} % ± {:.2f} %\n'.format(score_mem.mean(), score_mem.std()))
            print('\n')
        print('\n')

    print('Best accuracy obtained is {:.2f} % using {:} features and {:} channels'.format(max_acc * 100, k_feat, k_chan))
    print('all time', time.time() - tic)

