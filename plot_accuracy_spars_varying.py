import os.path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import stats, signal
from scipy.signal import savgol_filter

matplotlib.use('QT5agg')

spars = np.arange(1, 101, 1)
acc = np.array([67.9, 61.7, 63.0, 61.7, 66.7, 67.9, 66.7, 66.7, 65.4, 63.0, 67.9, 74.1, 58.0, 71.6, 70.4, 72.8, 71.6, 70.4, 71.6,
       72.8, 65.4, 66.7, 69.1, 70.4, 70.4, 72.8, 71.6, 65.4, 69.1, 67.9, 71.6, 69.1, 66.7, 66.7, 66.7, 65.4, 71.6, 71.6,
       76.5, 72.8, 69.1, 70.4, 67.9, 65.4, 71.6, 71.6, 69.1, 79.0, 77.8, 76.5, 74.1, 72.8, 74.1, 75.3, 76.5, 79.0, 77.8,
       76.5, 77.8, 76.5, 77.8, 74.1, 75.3, 75.3, 75.3, 77.8, 76.5, 76.5, 80.2, 82.7, 75.3, 79.0, 79.0, 81.5, 81.5, 82.7,
       80.2, 74.1, 77.8, 84.0, 85.2, 81.5, 80.2, 77.8, 74.1, 75.3, 76.5, 75.3, 77.8, 75.3, 77.8, 70.4, 75.3, 71.6, 65.4,
       64.2, 66.7, 69.1, 69.1, 74.1])


ax: matplotlib.pyplot.Axes
fig: matplotlib.pyplot.Figure
fig, ax = plt.subplots(1, 1, figsize=(11.6 / 1.8, 11.6 / 1.8 / np.sqrt(2)), tight_layout=True)

# kde = stats.gaussian_kde(acc, bw_method=0.01)
ax.plot(spars, acc, label='test-set accuracy', color='blue', marker='.', alpha=0.5)
ax.plot(spars, savgol_filter(acc, 100, 15), color='red', linestyle='--', label='smoothed test-set accuracy')

xx = np.linspace(0, 100, 1000)
# ax.plot(spars, kde(spars) * 4000, color='C3')


ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())

# ax.xaxis.grid(True, which='minor', linestyle=':', color='#e6e6e6', linewidth=0.6)
# ax.yaxis.grid(True, which='minor', linestyle=':', color='#e6e6e6', linewidth=0.6)
# ax.xaxis.grid(True, which='major', linestyle='-', color='#ebebeb', linewidth=1.2)
# ax.yaxis.grid(True, which='major', linestyle='-', color='#ebebeb', linewidth=1.2)
ax.xaxis.grid(True, which='minor', linestyle=':', color='#e6e6e6', linewidth=0.283, dashes=[0.7, 3])
ax.yaxis.grid(True, which='minor', linestyle=':', color='#e6e6e6', linewidth=0.283, dashes=[0.7, 3])
ax.xaxis.grid(True, which='major', linestyle='-', color='#ebebeb', linewidth=0.409, alpha=0.8)
ax.yaxis.grid(True, which='major', linestyle='-', color='#ebebeb', linewidth=0.409, alpha=0.8)
ax.set_xlim(-3, 103)
ax.set_axisbelow(True)
ax.set_xlabel(r'sparsity $\alpha_{k \%}$ (%)')
ax.set_ylabel('accuracy (%)')
ax.legend()

fig.savefig(os.path.join(os.getcwd(), 'figures', 'accuracy_spars_variation.svg'))
plt.show()
