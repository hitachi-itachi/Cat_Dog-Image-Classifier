import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os


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
IMG_SIZE = 50
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