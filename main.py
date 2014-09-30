# 12:59am - success!!
# new gets 92% f1 score; MultinomialNB() gets 96% :(

from scipy.io import loadmat
import numpy as np

mat = loadmat('spamTrain.mat')
X = mat['X']
y = mat['y']

ix = np.in1d(y.ravel(), 1).reshape(y.shape)
indices_spam = np.where(ix)[0]
X_spam = X[indices_spam]

ix = np.in1d(y.ravel(), 0).reshape(y.shape)
indices_ham = np.where(ix)[0]
X_ham = X[indices_ham]

# in X, each row is an email
# each column states whether a word exists in that email

class Old(object):
	def __init__(self):
		self.spam_trainers = np.log((X_spam.sum(axis = 0) + 1) / float(X_spam.shape[0])) # how to put this below and make it work? do it!
		self.ham_trainers = np.log((X_ham.sum(axis = 0) + 1) / float(X_ham.shape[0]))

	def calc(self, row, spam=True):
		new_email = X[row]
		n = np.where(new_email == 1)[0].shape[0] # X[row].sum() works too bc of 1 or 0; number words in both email and vocab list
		
		if spam:
			n2 = X_spam.shape[0] # number of spam emails in set
			trainer = self.spam_trainers
		else:
			n2 = X_ham.shape[0] # number of spam emails in set
			trainer = self.ham_trainers

		result = np.log(n2 / float(X.shape[0])) + (new_email * trainer).sum() # I sum bc I'm adding logs instead of multiplying
		return result

class New(object):
	def __init__(self):
		self.spam_trainers = np.log((X_spam.sum(axis = 0) + 1))
		self.ham_trainers = np.log((X_ham.sum(axis = 0) + 1))

	def calc(self, row, spam=True):
		new_email = X[row]
		n = np.where(new_email == 1)[0].shape[0] # X[row].sum() works too bc of 1 or 0; number words in both email and vocab list
		
		if spam:
			n2 = X_spam.shape[0] # number of spam emails in set
			trainer = self.spam_trainers
		else:
			n2 = X_ham.shape[0] # number of spam emails in set
			trainer = self.ham_trainers

		result = np.log(n2) + (new_email * trainer).sum() - np.log(X.shape[0]) - n * np.log(n2 + 1)
		return result

mat = loadmat('spamTest.mat')
X = mat['Xtest']
y = mat['ytest']

ix = np.in1d(y.ravel(), 1).reshape(y.shape)
indices_spam = np.where(ix)[0]
X_spam = X[indices_spam]

ix = np.in1d(y.ravel(), 0).reshape(y.shape)
indices_ham = np.where(ix)[0]
X_ham = X[indices_ham]

def test(old=True):
	if old:
		calc = Old().calc
	else:
		calc = New().calc

	y_pred = []

	for i in range(X.shape[0]): # all the emails, spam and ham
		spam_result = calc(i, spam=True)
		ham_result = calc(i, spam=False)

		if spam_result > ham_result:
			y_pred.append(1)
		else:
			y_pred.append(0)

	y_pred = np.array(y_pred)

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

test(old=False)