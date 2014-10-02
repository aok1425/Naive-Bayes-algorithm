# 12:59am - success!!
# new gets 93% f1 score; MultinomialNB() gets 96% :(
# why?

from scipy.io import loadmat
import numpy as np

# in X, each row is an email
# each column states whether a word exists in that email

class MyNaiveBayes(object):
	def fit(self, X, y):
		ix = np.in1d(y, 1).reshape(y.shape)
		indices_spam = np.where(ix)[0]
		self.X_spam = X[indices_spam]

		ix = np.in1d(y, 0).reshape(y.shape)
		indices_ham = np.where(ix)[0]
		self.X_ham = X[indices_ham]

		self.X = X

		self.spam_trainers = np.log((self.X_spam.sum(axis = 0) + 1))
		self.ham_trainers = np.log((self.X_ham.sum(axis = 0) + 1))

	def predict(self, X):
		y_pred = []

		for row in X:
			if self.calc(row, spam=True) > self.calc(row, spam=False):
				y_pred.append(1)
			else:
				y_pred.append(0)

		return np.array(y_pred)

	def calc(self, row, spam=True):
		n = np.where(row == 1)[0].shape[0] # X[row].sum() works too bc of 1 or 0; number words in both email and vocab list
		
		if spam:
			n2 = self.X_spam.shape[0] # number of spam emails in set
			trainer = self.spam_trainers
		else:
			n2 = self.X_ham.shape[0] # number of spam emails in set
			trainer = self.ham_trainers

		result = np.log(n2) + (row * trainer).sum() - np.log(self.X.shape[0]) - n * np.log(n2 + 1)
		return result

def score(y, y_pred):
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