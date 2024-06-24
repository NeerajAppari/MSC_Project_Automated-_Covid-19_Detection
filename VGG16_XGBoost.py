# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:06:35 2023

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
from tensorflow.keras.applications.vgg16 import VGG16

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

#Split data 
x_train, y_train, x_test, y_test = train_images,train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#Load model 
VGG_model = VGG16(weights='imagenet', include_top=False,input_shape=(SIZE, SIZE, 3))

#Make loaded layers as non-trainable. 
for layer in VGG_model.layers:
    layer.trainable = False
 
VGG_model.summary()

# convolutional network for RF
feature_extractor=VGG_model.predict(x_train)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)
X_for_training = features 

#XGBOOST
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_for_training, y_train)
import pickle

with open('C:/caar/College/AI/Pretrained/saved/vgg16_xgboost.pkl', 'wb') as model_file:pickle.dump(model, model_file)

#model.save('C:/caar/College/AI/Pretrained/saved/vgg16_xgboost.keras')

#Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

# predict using RF model. 
prediction = model.predict(X_test_features)

#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

#accuracy
from sklearn import metrics
#from sklearn.metrics import classification_report
#from sklearn.metrics import average_precision_score
print ("Accuracy = ", metrics.accuracy_score(test_labels,prediction))
#print (classification_report(test_labels, prediction))
#print ("Precision = ", average_precision_score('fish', 'phone'))

#Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, prediction)
#print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) 


#Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction = model.predict(input_img_features)[0] 
prediction = le.inverse_transform([prediction]) 

#Reverse the label encoder to original name
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", test_labels[n])
