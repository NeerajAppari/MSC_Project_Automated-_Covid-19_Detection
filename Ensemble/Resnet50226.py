# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:09:57 2024

@author: appar
"""

import numpy as np
import matplotlib.pyplot as plt 
import glob
import cv2 
import os
import seaborn as sns


from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import (
 
BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Conv2D, Dense
)
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# Read input images
print(os.listdir("C:/caar/College/AI/Pretrained"))


SIZE = 224
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
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded



# Normalize pixel values to between 0 and 1 
x_train, x_test = x_train / 255.0, x_test / 255.0

################################################## ####
from tensorflow.keras.utils import to_categorical 
y_train_one_hot = to_categorical(y_train) 
y_test_one_hot = to_categorical(y_test)

resnet50_model = Sequential()


pretrained_model1= tf.keras.applications.ResNet50(include_top=False,
input_shape=(SIZE,SIZE,3), pooling='avg',classes=2, weights='imagenet')
for layer in pretrained_model1.layers: layer.trainable=False

resnet50_model.add(pretrained_model1)


resnet50_model.add(Flatten()) 
resnet50_model.add(Dense(512, activation='relu')) 
resnet50_model.add(Dense(2, activation='softmax'))

resnet50_model.summary()


resnet50_model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
 
#history = model.fit(lung_train, infect_train, epochs = 20, validation_data = (lung_test, infect_test))


#epochs=10
history=resnet50_model.fit(x_train, y_train_one_hot, epochs=50, validation_data = (x_test, y_test_one_hot))

import pickle

with open('C:/caar/College/AI/Ensemble/Saved/resnet50.pkl', 'wb') as model_file:pickle.dump(history, model_file)

resnet50_model.save('C:/caar/College/AI/Ensemble/Saved/resnet50_1.hdf5')
resnet50_model.save('C:/caar/College/AI/Ensemble/Saved/resnet50_2.keras')

 
