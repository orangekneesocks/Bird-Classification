import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import pickle
import datetime
from sklearn.model_selection import train_test_split
#from sklearn.utils import class_weight

#import data:
X_infile = open('X.pickle', 'rb')
y_infile = open('y.pickle', 'rb')
X = pickle.load(X_infile)
y = pickle.load(y_infile)

#normalize images
X_norm = tf.keras.utils.normalize(X, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_norm,y,random_state=1)

poss_nodes = [256]
poss_dense = [5]

for num_nodes in poss_nodes:
    for num_layers in poss_dense:
        NAME = 'bird_clf_{}_dense_{}'.format(num_layers,num_nodes)
        tensorboard = TensorBoard(log_dir='bird_logs\{}'.format(NAME), histogram_freq=1)

        model = tf.keras.models.Sequential()
        model.add(Flatten())
        for l in range(num_layers):
            model.add(Dense(num_nodes, activation=tf.nn.relu))
        model.add(Dense(3, activation=tf.nn.softmax))

        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit(X_train, y_train, epochs = 8, validation_split=0.2, callbacks=[tensorboard])