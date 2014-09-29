from scipy.io import loadmat
import numpy as np

mat = loadmat('spamTest.mat')
X = mat['Xtest']
y = mat['ytest']
# X = mat['X']
# y = mat['y']

ix = np.in1d(y.ravel(), 1).reshape(y.shape)
indices_spam = np.where(ix)[0]
X_spam = X[indices_spam]

ix = np.in1d(y.ravel(), 0).reshape(y.shape)
indices_ham = np.where(ix)[0]
X_ham = X[indices_ham]

N = X.shape[0]

# in X, each row is an email
# each column states whether a word exists in that email
 
# (1 - n) log(n) + log(#word1 + 1) + ... + log(#wordn + 1) - log(N)
# n is # of spam emails; N is total # of emails

spam_trainers = X_spam.sum(axis = 0) + 1
ham_trainers = X_ham.sum(axis = 0) + 1

def spam(row):
	n = X_spam.shape[0]

	new_email = X[row]
	result = (1 - n) * np.log(n) + (new_email * spam_trainers).sum() - np.log(N)
	return result

def ham(row):
	n = X_ham.shape[0]

	new_email = X[row]
	result = (1 - n) * np.log(n) + (new_email * ham_trainers).sum() - np.log(N)
	return result

tot_words = np.where(X_spam.sum(axis=0)==1) + np.where(X_ham.sum(axis=0)==1)

def spam(row):
	n = X_spam.shape[0]
	denom = X_spam.sum(axis = 0)
	# word_count_of_spam = np.where(X_spam.sum(axis=0)==1)
	word_count_of_spam = np.log(X_spam.sum(axis = 0) + 1)

	return word_count_of_spam - np.log(tot_words + X.shape[1])

	# return ((((X_spam.sum(axis=0) + 1) * X[row])/float(n)).sum()) * n / float(N)
	return (np.log(X_spam.sum(axis=0)+1)*X[row]).sum()+np.log(n)-n*np.log(n)-np.log(N)

def ham(row):
	n = X_ham.shape[0]
	# return ((((X_ham.sum(axis=0) + 1) * X[row])/float(n)).sum()) * n / float(N)
	return (np.log(X_ham.sum(axis=0)+1)*X[row]).sum()+np.log(n)-n*np.log(n)-np.log(N)

def calc(row):
	print 'P that this new email is spam is {}; ham is {}'.format(spam(row), ham(row))
# overfitted

spams = []
hams = []

for i in range(N):
	spam_result = spam(i)
	ham_result = ham(i)

	if spam_result > ham_result:
		spams.append(i)
	else:
		hams.append(i)

print len(spams), len(indices_spam)
print len(hams), len(indices_ham)