from sklearn.naive_bayes import MultinomialNB
from scipy.io import loadmat
import numpy as np

mat = loadmat('spamTrain.mat')
X = mat['X']
y = mat['y'].ravel()

gnb = MultinomialNB()
y_pred = gnb.fit(X, y)

mat = loadmat('spamTest.mat')
X = mat['Xtest']
y = mat['ytest'].ravel()

y_pred = y_pred.predict(X)
print "Number of mislabeled points : {}".format((y != y_pred).sum())

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

for each in np.where(y_pred==1)[0]:
	if each in np.where(y==1)[0]:
		true_pos += 1
	else:
		false_pos += 1

for each in np.where(y_pred==0)[0]:
	if each in np.where(y==0)[0]:
		true_neg += 1
	else:
		false_neg += 1
		
print true_pos, false_pos
print false_neg, true_neg

precision = true_pos / float(true_pos + false_pos)
recall = true_pos / float(true_pos + false_neg)

print 'out of all predicted spam emails, {:.0%} were actually spam (precision)'.format(precision)
print 'out of all the actual spam emails, {:.0%} we predicted as being spam (recall)'.format(recall)

f1_score = 2 * precision * recall / (precision + recall)
print 'f1 score is {}'.format(f1_score)