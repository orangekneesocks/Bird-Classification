import pickle
import matplotlib.pyplot as plt

infile = open('X.pickle','rb')
X = pickle.load(infile)
infile2 = open('y.pickle','rb')
y = pickle.load(infile2)

print(y[1000:])
# imgnum = 1
# img = X[imgnum]
# label = y[imgnum]
# plt.imshow(img)
# plt.title(label)
# plt.show()