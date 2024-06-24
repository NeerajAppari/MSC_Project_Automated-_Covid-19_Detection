# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:55:31 2024

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
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import seaborn as sns
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Read input images 
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

#S

num_samples, height, width, channels = train_images.shape
X1 = train_images.reshape(num_samples, height * width * channels)

# Normalize the feature matrix if needed
X1_normalized = X1 / 255.0  # Example normalization to [0, 1] range


num_samples, height, width, channels = test_images.shape
X2 = test_images.reshape(num_samples, height * width * channels)

# Normalize the feature matrix if needed
X2_normalized = X2 / 255.0

#x_train, x_test, y_train, y_test = train_test_split(X_normalized1,train_labels_encoded)
x_train, y_train, x_test, y_test = X1_normalized,train_labels_encoded, X2_normalized, test_labels_encoded

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the SVM classifier
svm_classifier.fit(x_train, y_train)
#Now predict using the trained RF model. 


# Predict the labels for test set
y_pred = svm_classifier.predict(x_test)
y_pred = svm_classifier.predict(x_test) #Inverse le transform to get original label back. 
y_pred  = le.inverse_transform(y_pred )


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn import metrics
#from sklearn.metrics import classification_report #from sklearn.metrics import average_precision_score
print ("Accuracy = ", metrics.accuracy_score(test_labels, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, y_pred) 
print(cm)
sns.heatmap(cm, annot=True)

n=4 #Select the index of image to be loaded for testing 
img = X2_normalized[n]
plt.imshow(img)
 
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
prediction = np.argmax(svm_classifier.predict(input_img)) #argmax to convert categorical back to original
prediction = le.inverse_transform([prediction]) #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction) 
print("The actual label for this image is: ", test_labels[n])
 