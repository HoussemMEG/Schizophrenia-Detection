from statistics import NormalDist

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import poisson
from scipy import stats
matplotlib.use('QT5agg')

best_val = 78.4  # before any optimization: 78.4  /  after optimisation: 82.7  /  after outlier removal: 84.6
val = [
    [70.4, 70.4, 70.4, 72.8, 71.6, 74.1, 72.8, 67.9, 70.4, 65.4, 70.4, 71.6, 63.0, 69.1, 69.1, 70.4, 72.8, 71.6, 65.4,
     70.4, 69.1, 67.9, 71.6, 67.9, 70.4, 71.6, 69.1, 72.8, 66.7, 71.6, 69.1, 67.9, 67.9],
    [72.8, 69.1, 71.6, 71.6, 67.9, 72.8, 67.9, 69.1, 76.5, 71.6, 67.9, 71.6, 70.4, 69.1, 63.0, 70.4, 70.4, 67.9, 66.7,
     70.4, 71.6, 66.7, 70.4, 77.8, 70.4, 69.1, 70.4, 71.6, 72.8, 70.4, 75.3, 76.5, 71.6],
    [69.1, 69.1, 72.8, 71.6, 70.4, 67.9, 67.9, 66.7, 69.1, 75.3, 71.6, 71.6, 69.1, 72.8, 77.8, 74.1, 71.6, 72.8, 69.1,
     70.4, 66.7, 70.4, 74.1, 71.6, 72.8, 74.1, 65.4, 71.6, 69.1, 71.6, 70.4, 64.2, 67.9],
    [70.4, 72.8, 70.4, 70.4, 74.1, 72.8, 74.1, 69.1, 69.1, 70.4, 66.7, 72.8, 65.4, 69.1, 71.6, 72.8, 67.9, 70.4, 69.1,
     72.8, 71.6, 69.1, 71.6, 70.4, 70.4, 76.5, 71.6, 70.4, 71.6, 74.1, 69.1, 72.8, 66.7],
    [70.4, 71.6, 71.6, 77.8, 70.4, 70.4, 71.6, 67.9, 69.1, 71.6, 69.1, 66.7, 70.4, 69.1, 71.6, 69.1, 66.7, 74.1, 71.6,
     65.4, 71.6, 71.6, 67.9, 69.1, 70.4, 71.6, 65.4, 67.9, 69.1, 67.9, 70.4, 70.4, 70.4],
    [67.9, 69.1, 65.4, 70.4, 75.3, 66.7],
    [74.1, 70.4, 64.2, 69.1, 69.1, 69.1],
    [70.4, 70.4, 66.7, 70.4, 70.4, 71.6],
    [67.9, 69.1, 74.1, 70.4, 72.8, 63.0],
    [67.9, 71.6, 72.8, 69.1, 69.1, 66.7],
]

val = [x for gg in val for x in gg]
val = np.array(val)
mu = val.mean()
sigma = val.std()

print('Number of repetition: {:}'.format(len(val)))
print('min = {:.1f} % \t max = {:.1f} %'.format(val.min(), val.max()))
print('val = {:.1f} % ± {:.1f} %'.format(val.mean(), val.std()))
print('\tp-value {:.1e}'.format(stats.norm.sf(best_val, loc=mu, scale=sigma)))

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
fig, ax = plt.subplots()
plt.plot(x, stats.norm.pdf(x, mu, sigma), color='red')
sns.histplot(val, bins=len(set(val)), kde=True, ax=ax, stat='density')
plt.show()
