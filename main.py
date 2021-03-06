# 12:59am - success!!
# new gets 93% f1 score; MultinomialNB() gets 96% :(
# why?

from scipy.io import loadmat
import numpy as np

# in X, each row is an email
# each column states whether a word exists in that email

class MyNaiveBayes(object):
	def fit(self, X, y):
		"""y needs to be 0-indexed, and there must not be gaps btwn 0 and max(y)."""
		self.categs = max(y) + 1 # num of rows
		emails = X.shape[1] # num of columns
		self.trainers = np.zeros(self.categs * emails).reshape(self.categs, emails)
		self.n2 = {} # num of emails of that categ amongst the training set; num of spam emails

		for i in range(self.categs): # cycling through the # of categs, like spam/ham
			ix = np.in1d(y, i).reshape(y.shape)
			indices_spam = np.where(ix)[0]
			indices = X[indices_spam]
			self.n2[i] = indices.shape[0]

			self.trainers[i] = np.log((indices.sum(axis = 0) + 1))

		self.X = X		

	def predict(self, X):
		y_pred = []

		for row in X:
			probabilities = [self.calc(row, categ) for categ in range(self.categs)] # for each categ, P that the row, or email, belongs to that categ
			max_index = probabilities.index(max(probabilities)) # index of the highest P
			y_pred.append(max_index)

		return np.array(y_pred)

	def calc(self, row, categ):
		"""Categ is 1 for spam and 0 for ham."""
		n = np.where(row == 1)[0].shape[0] # X[row].sum() works too bc of 1 or 0; number words in both email and vocab list
		n2 = self.n2[categ]
		trainer = self.trainers[categ]

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

	return f1_score