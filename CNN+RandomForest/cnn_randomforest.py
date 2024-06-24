# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:43:07 2024

@author: appar
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
 BatchNormalization, SeparableConv2D, MaxPooling2D, 
Activation, Flatten, Dropout, Conv2D, Dense
)
import os
import seaborn as sns


print(os.listdir("C:/caar/College/AI/Pretrained"))
SIZE = 256
#Channel=3 #Resize images
#lists

train_images = []
train_labels = [] 
for directory_path in glob.glob("C:/caar/College/AI/Pretrained/Train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path,"*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

#lists to arrays 
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# test/validation into lists
test_images = []
test_labels = [] 
for directory_path in glob.glob("C:/caar/College/AI/Pretrained/Test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path,"*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

#lists to arrays 
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data 
x_train, y_train, x_test, y_test = train_images,train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0



#############################
activation = 'sigmoid' 
feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (SIZE, SIZE, 3)))
feature_extractor.add(BatchNormalization())


feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
 
feature_extractor.add(BatchNormalization()) 
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())


feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization()) 
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())





#Now, let us use features from convolutional network for RF
X_for_RF = feature_extractor.predict(x_train) #This is out X input to RF


#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
 
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)


# Train the model on training data
RF_model.fit(X_for_RF, y_train) #For sklearn no one hot encoding
import pickle

with open('C:/caar/College/AI/Pretrained/CNN+RandomForest/Saved/cnn_randomforest.pkl', 'wb') as model_file:pickle.dump(RF_model, model_file)


#Send test data through same feature extractor process 
X_test_feature = feature_extractor.predict(x_test) #Now predict using the trained RF model. 
prediction_RF = RF_model.predict(X_test_feature) #Inverse le transform to get original label back. 
prediction_RF = le.inverse_transform(prediction_RF)

#Print overall accuracy 
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))
Accuracy = 0.55





#Confusion Matrix - verify accuracy of each class
 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, prediction_RF) #
print(cm)



#Check results on a few select images #n=5
n=9 #Select the index of image to be loaded for testing img = x_test[n]
plt.imshow(img)
 
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_features=feature_extractor.predict(input_img) 
prediction_RF = RF_model.predict(input_img_features)[0]
prediction_RF = le.inverse_transform([prediction_RF]) #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_RF) 
print("The actual label for this image is: ", test_labels[n])
