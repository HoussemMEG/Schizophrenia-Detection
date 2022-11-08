import json
import os

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from scipy import stats, interpolate
from collections import Counter

from utils import print_c

matplotlib.use('QT5agg')

file_name = ['LDA  test_LOO  rep_test_50  val_CV  rep_val_49 merge_test with_preselection.json',     ### finals
             'LDA  test_LOO  rep_test_50  val_CV  rep_val_49  without_preselection.json',
             'LDA  test_25  rep_test_1000  val_CV  rep_val_25.json',
             'LDA  test_25  rep_test_100000  val_CV  rep_val_25.json',
             'LDA  test_10  rep_test_1000  val_CV  rep_val_40.json',
             'LDA  test_10  rep_test_10000  val_CV  rep_val_40.json',
             ][-3]
path = os.getcwd()
print_c(file_name, 'green', bold=True)
file_name = os.path.join(path, '..', 'test memory', file_name)
with open(file_name, 'r') as file:
    data: dict = json.load(file)

models = ['CP1', 'CPz', 'CP2', 'Voted']  # , 'Merged'
# models = ['channel']


def compute_percentile(y, n, percentile, tol):
    cumsum = np.cumsum(y)  # cumulative sum (PMF)
    # p_idx = np.where(np.abs(cumsum - percentile) < tol)[-1]  # index where the integral == percentile (in xx)
    # p_idx = int(p_idx[round(len(p_idx) / 2)])  # middle of the indexes that fits the tol
    p_idx = np.abs(cumsum - percentile).argmin()
    p_val = p_idx / n  # rescale xx to 0 - 100 %
    return p_idx, p_val


def plot(acc, model, i):
    if i != 3:
        ax: plt.Axes = fig.add_subplot(gs[0, i])
    else:
        ax: plt.Axes = fig.add_subplot(gs[1, -1])

    # KDE
    kde = stats.gaussian_kde(acc, bw_method=0.5)
    if i == 3:
        kde = stats.gaussian_kde(acc, bw_method=0.4)

    # PLOT
    temp = np.sort(np.array(list(set(acc))))
    resolution = round(min(temp[1:] - temp[:-1]))
    bins = np.arange(0, 100 + 2 * resolution, resolution)
    ax.hist(acc, density=True, bins=bins, alpha=0.7, color='b', align='left')  # rwidth=20
    # ax.plot(x, kde(x), color='C3')

    values = [*range(0, round(min(acc)), resolution)]
    count = [0] * len(values)
    # values, count = [], []
    for key, val in Counter(acc).items():
        values.append(key)
        count.append(val)
    if max(values) < 100:
        repeat = (100 - max(values)) / resolution
    else:
        repeat = 0

    for i in range(int(repeat)):
        values.append(max(values) + (i+1) * resolution)
        count.append(0)
    n = 10

    f = interpolate.interp1d(values, count, kind='quadratic')
    x = np.linspace(0, 100, 100 * n)
    y = f(x) / (len(acc) * resolution)
    ax.plot(x, y, color='r', alpha=0.8)
    _, y_max = ax.get_ylim()
    # y_max = 0.07580327661513762
    # y_max = 0.04721961148342516
    ax.set_ylim(0, y_max)

    percentile, tol = 0.50, 5e-3  # percentile and the tol
    p_idx, p_val = compute_percentile(y / y.sum(), n, percentile, tol)
    ax.axvline(x=p_idx/n, color='k', linestyle=':', label='mean', ymax=y[p_idx] / y_max)  # mean vertical line

    percentile, tol = 0.05, 5e-3  # percentile and the tol
    p_idx, p_val = compute_percentile(y / y.sum(), n, percentile, tol)
    print_c("\t   {:}-percentile = {:.2f} %".format(round(100 * percentile), p_val), bold=True)
    # ax.axvline(x=p_idx / n, color='r', linestyle='--', label='mean', ymax=y[p_idx] / y_max)  # mean vertical line
    ax.fill_between(x[:p_idx], y[:p_idx], color="r", alpha=0.85)
    ax.text(x[p_idx]-20, y[p_idx]+0.002, '$5\!-\!th\,percentile$', fontsize='small')

    # percentile, tol = 0.95, 5e-3  # percentile and the tol
    # p_idx, p_val = compute_percentile(y / y.sum(), n, percentile, tol)
    # print_c("\t   {:}-percentile = {:.2f} %".format(round(100 * percentile), p_val), bold=True)
    # ax.axvline(x=(p_idx + 1) / n, color='r', linestyle='--', label='mean', ymax=y[p_idx] / y_max)  # mean vertical line
    # # ax.fill_between(x[p_idx:], y[p_idx:], color="r", alpha=0.4)

    # minor axis
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # ax.xaxis.grid(True, which='major', linestyle='-', color='#ebebeb', linewidth=1.2)
    # ax.yaxis.grid(True, which='major', linestyle='-', color='#ebebeb', linewidth=1.2)
    # ax.xaxis.grid(True, which='minor', linestyle=':', color='#e6e6e6', linewidth=0.6)
    # ax.yaxis.grid(True, which='minor', linestyle=':', color='#e6e6e6', linewidth=0.6)
    # ax.xaxis.grid(True, which='minor', linestyle=':', color='#e6e6e6', linewidth=0.283, dashes=[0.7, 3])
    # ax.yaxis.grid(True, which='minor', linestyle=':', color='#e6e6e6', linewidth=0.283, dashes=[0.7, 3])
    # ax.xaxis.grid(True, which='major', linestyle='-', color='#ebebeb', linewidth=0.409, alpha=0.8)
    # ax.yaxis.grid(True, which='major', linestyle='-', color='#ebebeb', linewidth=0.409, alpha=0.8)
    ax.set_xlim(27, 105)
    ax.set_axisbelow(True)
    ax.set_xlabel('$accuracy \%$')
    ax.set_ylabel('$density$')
    ax.set_title('${:}$'.format(model))


# Create 2x2 sub plots
fig = plt.figure(figsize=(11.6 / 1.2, 11.6 / 1.35 / np.sqrt(2)), tight_layout=False)
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.45, figure=fig)

for i, model in enumerate(models):
    print_c('   ' + model, 'blue', bold=True)
    for key, val in data[model].items():
        if key == 'merged':
            continue
        if key != 'param':
            val = np.sort(100 * np.array(val))
            mean, std = val.mean(), val.std()
            print_c('\t {:<5}:  mean {:4.1f} %    std {:4.1f} %'.format(key.capitalize(), mean, std), bold=True)
            if key == 'test':
                if model == 'CP1':
                    model = 'CP_1'
                elif model == 'CP2':
                    model = 'CP_2'
                elif model == 'CPz':
                    model = 'CP_z'
                else:
                    model = 'voted'
                plot(val, model, i)
        else:
            param = list(set(val))
            param.sort()
            for elem in param:
                print_c('\t\tParam: {:<2.2f} \t N = {:<3}'.format(elem, val.count(elem)))

val = data['Voted']['test']
contain = list(set(val))
contain.sort()
container = []
for elem in contain:
    container.append((100 * elem, val.count(elem)))
    print('{:<3.0f} %  N = {:}'.format(100 * elem, val.count(elem)))
fig.savefig('../figures/densities.svg')
plt.show()

"""
'LDA  test_LOO  rep_test_50  val_CV  rep_val_49  without_preselection.json',       # 0
             'LDA  test_LOO  rep_test_50  val_CV  rep_val_49.json',                             # 1
             'LDA  test_10  rep_test_500  val_10  rep_val_400.json',                            # 2   ---
             'LDA  test_25  rep_test_200  val_CV  rep_val_25.json',                             # 3   --
             'LDA  test_10  rep_test_2000  val_10  rep_val_400.json',                           # 4
             'LDA  test_10  rep_test_500  val_CV  rep_val_40.json',                             # 5   ---
             'LDA  test_25  rep_test_100000  val_CV  rep_val_25.json',                          # 6
             'LDA  test_BLOO  rep_test_1  val_0  rep_val_50.json',                              # 7
             'LDA  test_LOO  rep_test_50  val_CV  rep_val_49  merge_test_True.json',
             'LDA  test_LOO  rep_test_50  val_CV  rep_val_49.json',
             'LDA  test_LOO  rep_test_50  val_CV  rep_val_49  merge_test_True channel CP1 all 2% pars level.json',
             'LDA  test_LOO  rep_test_50  val_CV  rep_val_49  merge_test_True channel CPz all 2% pars level.json',
             'LDA  test_LOO  rep_test_50  val_CV  rep_val_49  merge_test_True channel CP2 all 2% pars level.json',
             'LDA  test_LOO  rep_test_50  val_CV  rep_val_49  merge_test_True channel CP2 all 1% pars level.json',
             'LDA  test_LOO  rep_test_50  val_CV  rep_val_49 all 2% but CP2 in 1%.json',
             'LDA  test_LOO  rep_test_50  val_CV  rep_val_49 all 2%.json',
             'LDA  test_LOO  rep_test_50  val_CV  rep_val_49  merge_test_Truechannel.json',   
"""
