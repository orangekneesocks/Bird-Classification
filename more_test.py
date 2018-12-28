#0: duck, 1:hummingbird, 2:owl

import pickle
import matplotlib.pyplot as plt
import numpy as np

infile = open('X.pickle','rb')
X = pickle.load(infile)
infile2 = open('y.pickle','rb')
y = pickle.load(infile2)

num_img = 1500
img1 = X[num_img]
img_title = y[num_img]
plt.imshow(img1)
plt.title(img_title)
plt.show()

# print(X.shape)
# print(len(y))