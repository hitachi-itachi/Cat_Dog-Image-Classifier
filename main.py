import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
import theano

DATADIR = "C:\Python project\Dog_and_Catclassifierproject\\train"
CATEGORIES  = ['CAT','DOG'] #we can use categories to map out which one is the cat and which one is the dog using array position.

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #path to cats or dogs dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # first argument = read the file in specificed path, second is to read it in grayscale.rgb is 3 times the size of greyscale
        plt.imshow(img_array, cmap = "gray")
        plt.show()
        break
    break

print(img_array.shape) #we have to make everything the same shape

#sizing of the image
#some of the focus on the image might be smaller so we have to becareful on how we set the image size
IMG_SIZE = 200 #shaping image to the same size.
new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
plt.imshow(new_array,cmap='gray')
plt.show()

training_data = []
name_data = []

#we need to map things to a numerical value
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path to cats or dogs dir
        class_num = CATEGORIES.index(category) #making it 0 = cat, 1 = dog
        for img in os.listdir(path): #label about each file name that is inside the subfolder :0
            try:
                img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)  # rgb is 3 times the size of greyscale
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num]) #our training_data has two array which allow for loop to get both featurs and label!
                name_data.append(img)
            except Exception as e:
                pass



create_training_data()

#its very important that we wanna have a 50/50 spilt between data 50% cat and 50% dog
print(len(training_data))

import random
random.shuffle(training_data)

#Loop thru for 10 elements of training_data, take second element of the list. If cat = 0 if dog = 1
#for sample in training_data[:10]:
#print(sample[0]) #basically printing the second argument of the training data since it has two argument


# Normalising X and converting labels to categorical data
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

batch_size = 16
nb_classes =4
nb_epochs = 6 # previous epoch was 5 increase to 6 to see if it has higher accuracy
img_rows, img_columns = IMG_SIZE, IMG_SIZE # Was 200,200 I changed to 50
img_channel = 1 #was three I changed to 1
nb_filters = 32
nb_pool = 2
nb_conv = 3

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(IMG_SIZE, IMG_SIZE, 1)), #was 3 I changed to 1 and 200 then changed to 50 as well
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(4,  activation=tf.nn.softmax)
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size = batch_size, epochs = nb_epochs, verbose = 1, validation_data = (X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose = 0 )
print("Test Score: ", score[0])
print("Test accuracy: ", score[1])

""""for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE,IMG_SIZE, 1)  #the last 1 is cuz its a grayscale

import pickle
pickle_out = open("x.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open("x.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
x = pickle.load(pickle_in)

print(x[1])

"""