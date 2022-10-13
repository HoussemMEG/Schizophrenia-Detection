import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

n = 1000000
w = 0.2
X = np.random.uniform(-1, 1, size=(n, 1))
Y = np.hstack((np.ones(int(w * n)), np.zeros(int((1-w) * n))))

# taking the priors into account
clf = LinearDiscriminantAnalysis(solver='svd', priors=[0.5, 0.5], tol=0.0001, covariance_estimator=None)
clf.fit(X, Y)
print('Taking the priors into account: score= {:.2f} %'.format(100 * clf.score(X, Y)))

# let the algorithm take care about the priors
clf = LinearDiscriminantAnalysis(solver='svd', priors=None, tol=0.0001, covariance_estimator=None)
clf.fit(X, Y)
print('Algorithm decides for us: score= {:.2f} %'.format(100 * clf.score(X, Y)))
