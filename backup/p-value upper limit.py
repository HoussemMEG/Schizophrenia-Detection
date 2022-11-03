import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=[0.5, 0.5], n_components=None, tol=0.0001)
# clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, tol=0.0001)

n = 81
balance = 49 / 81
# balance = 0.1
repeat = 20 * 64 * 30
# repeat = 10000

print(balance)
max_val = 0
for i in tqdm(range(repeat)):
    SZ = np.random.uniform(-1, 1, size=(int(balance * n)))
    y_SZ = np.ones_like(SZ)
    CTL = np.random.uniform(-1, 1, size=(int((1 - balance) * n)))
    y_CTL = np.zeros_like(CTL)
    X = np.concatenate((SZ, CTL))[:, np.newaxis]
    y = np.concatenate((y_SZ, y_CTL))
    clf.fit(X, y)
    score = clf.score(X, y) * 100
    print(f'{i = }\t{score = :.2f}')
    if score > max_val:
        max_val = score
print('\t\tMax score obtained: {:.2f}'.format(max_val))
