import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np

from pandas import read_csv
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

DATADIR = "C:\Python project\Dog_and_Catclassifierproject\\train"
CATEGORIES  = ['CAT','DOG'] #we can use categories to map out which one is the cat and which one is the dog using array position.

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #path to cats or dogs dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #rgb is 3 times the size of greyscale
        plt.imshow(img_array, cmap = "gray")
        plt.show()
        break
    break

print(img_array.shape) #we have to make everything the same shape

#sizing of the image
#some of the focus on the image might be smaller so we have to becareful on how we set the image size
IMG_SIZE = 50 #shaping image to the same size.
new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
plt.imshow(new_array,cmap='gray')
plt.show()

training_data = []

#we need to map things to a numerical value
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)  # rgb is 3 times the size of greyscale
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

#its very important that we wanna have a 50/50 spilt between data 50% cat and 50% dog
print(len(training_data))

import random
random.shuffle(training_data)

#Loop thru for 10 elements of training_data, take second element of the list. If cat = 0 if dog = 1
for sample in training_data[:10]:
    print(sample[1])

x = training_data
y = ['CAT','DOG']


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.20, random_state=1)




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