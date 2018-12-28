import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle

X_infile = open('X.pickle','rb')
y_infile = open('y.pickle','rb')
X = pickle.load(X_infile)
y = pickle.load(y_infile)

X_scale = X/255.0
num_imgs = X_scale.shape[0]
X_scale = X_scale.reshape(num_imgs,200,200,1)

nodes = 64
convlayers = 2

NAME = 'bird_cnn_{}_convlayers_{}_nodes'.format(nodes, convlayers)
tensorboard = TensorBoard(log_dir='bird_logs\{}'.format(NAME))

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = (200,200,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Dense(3, activation = 'softmax'))

model.compile(
	loss='sparse_categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])
model.fit(X_scale, y, epochs=10, validation_split=.2, callbacks=[tensorboard])

# notes: val acc never seems to go higher than 50, if even
# probably end the project here