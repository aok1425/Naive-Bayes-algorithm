# 12:59am - success!!=

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

spam_trainers = np.log((X_spam.sum(axis = 0) + 1) / float(X_spam.shape[0]))
ham_trainers = np.log((X_ham.sum(axis = 0) + 1) / float(X_ham.shape[0]))

# maybe the n in the trainers and the n during calculation are different?

def spam(row):
	# another big change here is assuming that when i'm calc'g a new email,
	# i only care about the words that exist in the email. only those words
	# count towards n. all the words in the vocab don't count towards n.
	new_email = X[row]
	n = np.where(new_email == 1)[0].shape[0]

	result = np.log(X_spam.shape[0] / float(X.shape[0])) + (new_email * spam_trainers).sum() # I sum bc I'm adding logs instead of multiplying
	# b4, i had n*log(n). but the 1st n refers to the # of words, which is local
	# the 2nd n refers to how likely a word appears in an email out of ALL SPAM EMAILS, so it's not local

	# i only care about the words that exist in the new email, and by * 1, i get only those
	# the / n refers to the P that a word appears given that an email is spam, so we care
	# about all the spam emails, not specific to this email

	return result

def ham(row):
	# before i made the 0s in new_email 1s, to count the hams
	# but now i think, does email contain certain words? then, 
	# how likely do hams contain these words?
	# if i changed 1 to 0, that wld be, how likely do hams contain words that
	# don't exist in this email
	new_email = X[row]
	n = np.where(new_email == 1)[0].shape[0]

	result = np.log(X_ham.shape[0] / float(X.shape[0])) + (new_email * ham_trainers).sum()
	return result

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