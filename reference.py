from sklearn.naive_bayes import GaussianNB
from scipy.io import loadmat
import numpy as np

mat = loadmat('spamTest.mat')
X = mat['Xtest']
y = mat['ytest'].ravel()

gnb = GaussianNB()
y_pred = gnb.fit(X, y).predict(X)
print "Number of mislabeled points : {}".format((y != y_pred).sum())

print len(np.where(y_pred==1)[0]), len(np.where(y==1)[0])
print len(np.where(y_pred==0)[0]), len(np.where(y==0)[0])
"""
for each in np.where(y_pred==1)[0]:
	if each not in np.where(y==1)[0]:
		print each, 'was in y_pred, not in y'

for each in np.where(y==1)[0]:
	if each not in np.where(y_pred==1)[0]:
		print each, 'was in y, not in y_pred'"""