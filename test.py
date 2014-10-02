from main import *

def test(old=True):
	if old:
		calc = Old().calc
		alg = Old()
	else:
		calc = New().calc
		alg = New()

	mat = loadmat('spamTrain.mat')
	X = mat['X']
	y = mat['y'].ravel()

	alg.fit(X, y) # like scikit-learn, I should only accept y in ravel() form

	mat = loadmat('spamTest.mat')
	X = mat['Xtest']
	y = mat['ytest'].ravel()

	y_pred = alg.predict(X)

	score(y, y_pred)

test(old=True)
print ''
test(old=False)