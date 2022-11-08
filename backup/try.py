import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.svm import LinearSVC

# n = 10
# pars = 0.80
# for elem in np.arange(pars-0.01*n, pars+0.01*n, 0.01):
#     print('{:.2f},'.format(elem))

X = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7]]).T
y = np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

clf = LinearSVC(tol=1e-2, class_weight='balanced', max_iter=1000)
clf.fit(X, y)
print(clf.score(X, y))
