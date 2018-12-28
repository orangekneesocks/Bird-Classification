#foxglove code
#0: duck, 1:hummingbird, 2:owl

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle

datadir = r'C:\Users\tdhar\Documents\Personal Projects\birds\dataset'
categories = ['duck_birds','hummingbird','owl']

img_size = 200

#cycle through files and put in training data
training_data = []
def create_training_data():
    for category in categories:
        class_num = categories.index(category)
        path = os.path.join(datadir, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resize_img_array = cv2.resize(img_array,(img_size,img_size))
                training_data.append([resize_img_array,class_num])
            except Exception as e:
                pass

create_training_data()

training_data = np.array(training_data)

X = []
y = []

for feature, label in training_data:
    X.append(feature)
    y.append(label)

X = np.array(X).reshape(-1,img_size,img_size)
y = np.array(y).reshape(-1,)

pickle_out = open('X.pickle','wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle','wb')
pickle.dump(y, pickle_out)
pickle_out.close()