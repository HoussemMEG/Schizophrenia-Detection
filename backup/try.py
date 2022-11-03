import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

# n = 10
# pars = 0.80
# for elem in np.arange(pars-0.01*n, pars+0.01*n, 0.01):
#     print('{:.2f},'.format(elem))

X = np.array([[0, 1, 2, 3, 4, 5, 6, 7]]).T
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


f_test = LeaveOneOut()
f_test = StratifiedKFold(n_splits=2)

f_validation = LeaveOneOut()
f_validation = StratifiedKFold(n_splits=2)

for idx_learning, subj_test in f_test.split(np.zeros_like(X), y):
    print('learning', idx_learning, 'test', subj_test)

    for idx_train, subj_validation in f_validation.split(idx_learning, y[idx_learning]):
        print('\ttrain', idx_learning[idx_train], 'validation', idx_learning[subj_validation])
    print('')

# mean 84.010889292196
gg = [80.7017543859649, 82.45614035087719, 82.75862068965517, 84.48275862068965, 89.65517241379311]
n = [15, 15, 14, 14, 14]
gg = [80.7017543859649, 82.45614035087719, 82.75862068965517, 84.48275862068965, 89.65517241379311]
gg = [80.7017543859649, 82.45614035087719, 82.75862068965517, 90.51724137931035, 96.05911330049263]
gg = [100, 100, 100, 100, 100]
res = np.array([gg[0] * 15, gg[1] * 15, gg[2] * 14, gg[3] * 14, gg[4] * 14])
print(res)
print(res.sum() / 72)
# print(np.array(gg).mean())