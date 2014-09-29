from sklearn.naive_bayes import GaussianNB
from scipy.io import loadmat
import numpy as np

mat = loadmat('spamTrain.mat')
X = mat['X']
y = mat['y'].ravel()

gnb = GaussianNB()
y_pred = gnb.fit(X, y)

mat = loadmat('spamTest.mat')
X = mat['Xtest']
y = mat['ytest'].ravel()

y_pred = y_pred.predict(X)
print "Number of mislabeled points : {}".format((y != y_pred).sum())

print len(np.where(y_pred==1)[0]), len(np.where(y==1)[0])
print len(np.where(y_pred==0)[0]), len(np.where(y==0)[0])

count = 0
for each in np.where(y_pred==1)[0]:
	if each not in np.where(y==1)[0]:
		count += 1

print count, 'was in y_pred, not in y'

count = 0

for each in np.where(y==1)[0]:
	if each not in np.where(y_pred==1)[0]:
		count += 1
		
print count, 'was in y, not in y_pred'