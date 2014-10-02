from main import *

def test():
	mat = loadmat('spamTrain.mat')
	X = mat['X']
	y = mat['y'].ravel()

	alg = MyNaiveBayes()
	alg.fit(X, y) # like scikit-learn, I should only accept y in ravel() form

	mat = loadmat('spamTest.mat')
	X = mat['Xtest']
	y = mat['ytest'].ravel()

	y_pred = alg.predict(X)

	score(y, y_pred)

test()