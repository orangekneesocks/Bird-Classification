import pickle
import numpy as np

X_infile = open('X.pickle','rb')
y_infile = open('y.pickle','rb')
X = pickle.load(X_infile)
y = pickle.load(y_infile)

X_cnn = []
for i in range(X.shape[0]):
	img_cnn = [i,X[i]]
	X_cnn.append(img_cnn)

print(X_cnn[3])