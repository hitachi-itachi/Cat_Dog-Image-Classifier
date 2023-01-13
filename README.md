# Cat_Dog-Image-Classifier



## What is this project about?
This project evaluates the ability of CNN(Convulutional neural network) to make accurate prediction on a cats and dogs dataset. I 

## Application Code
### Main Framework used(for deeplearning):
Tensor & Keras for curating the CNN model <br />
Scikit-learn for trainining and testing <br />

### Library used:
Numpy <br />
Opencv <br />
os <br />

### External link:
Cat & Dog Data set: 

## Data Preprocessing

### Converting image to greyscale
![bandicam 2023-01-13 21-00-24-600](https://user-images.githubusercontent.com/64311133/212325908-c7d2e5d9-815e-4f76-8487-bfbe584f76eb.jpg)

Basically this code converts my image to greyscale.So it would be less computationally taxing and simplify the algorithm needed for this project.

### Making sure the Image are equal size:


![bandicam 2023-01-13 21-17-02-480](https://user-images.githubusercontent.com/64311133/212332937-5a65fc06-0559-452a-bc1a-be2215a53e49.jpg)
All Image are of different sizes we have to make sure that all image are of equal sizes to eliminate any abnormalites that may show up during the training/testing phase.

### Original Image:

![cat 0](https://user-images.githubusercontent.com/64311133/212333450-d37393f3-bed8-444b-9ffb-68caf33dcf18.jpg)

### Image after prepocessing
![bandicam 2023-01-08 16-42-03-505](https://user-images.githubusercontent.com/64311133/212334115-7c1a9331-a957-4e34-b4dc-21072865caa7.jpg)

## Creation of training data
![bandicam 2023-01-13 21-50-31-960](https://user-images.githubusercontent.com/64311133/212335457-4f592c5d-57f1-46ff-838c-68ec12db95f3.jpg)
This code is basically use to create training data to use for train/test spilt 

## Randomize the training data
![bandicam 2023-01-13 22-08-04-631](https://user-images.githubusercontent.com/64311133/212339889-aca7e023-3ab2-4112-8e1d-97cfaffd5d12.jpg) </br>
The training data has to be randomize because if not it would be the first half of the data is dog and the other half is cat which may cause overfitting.

After that I print out a sample of the output to make sure that the cat and dog data is well mixed. 0 = cat and 1 = dog

###Sample output: </br>
![bandicam 2023-01-13 22-08-14-312](https://user-images.githubusercontent.com/64311133/212340666-8b06c3b6-4e0f-49a6-88c2-4d1127e1cedd.jpg)

## Curated the CNN Model and send the data for prediction
![bandicam 2023-01-13 22-17-30-591](https://user-images.githubusercontent.com/64311133/212342935-a3c7d09a-93f7-4415-a355-8326f81144ad.jpg)

###What we are using is LeNet-5 CNN architcture:

CNN basically aims to reduce the amount of features present in a data set. 
Three 3 main layers convolution layer, pooling layer, fully connected layer.

Convolution layer: Extract varioues feature from image and form feature map
Pooling layer: Extract the biggest element from the feature map
Fully connected: Where layers start to complete and prediction starts to take place </br>

### An image of LeNet-5 Architecture

![1lvvWF48t7cyRWqct13eU0w](https://user-images.githubusercontent.com/64311133/212344983-c2a314bf-c612-42f8-b1aa-bee32a1a608e.jpeg)


