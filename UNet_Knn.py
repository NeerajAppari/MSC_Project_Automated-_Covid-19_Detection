# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:10:27 2024

@author: appar
"""

import tensorflow as tf 
import os
import random
 
import numpy as np 
from tqdm import tqdm
from skimage.io import imread, imshow 
from skimage.transform import resize 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt 
import glob
import cv2


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Conv2D, Dense
)
import os
import seaborn as sns
 
seed = 42 
np.random.seed = seed

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3



train_images = [] 
train_labels = []
for directory_path in glob.glob("C:/caar/College/AI/Pretrained/Train/*"):
    label = directory_path.split("\\")[-1] 
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        train_images.append(img) 
        train_labels.append(label)
 


train_images = np.array(train_images) 
train_labels = np.array(train_labels)



# test 
test_images = [] 
test_labels = []
for directory_path in glob.glob("C:/caar/College/AI/Pretrained/Test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        test_images.append(img) 
        test_labels.append(fruit_label)

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




#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
 
#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)


c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)


c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)


c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
 
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)


c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)


#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2,2)	, padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)


u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
 
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)


u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)


u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)


model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

feature_extractor=model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)


X_for_KNN = features

from sklearn.ensemble import BaggingClassifier 
from sklearn.neighbors import KNeighborsClassifier
KNN_model = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)


# Train the model on training data
 
KNN_model.fit(X_for_KNN,y_train)

import pickle

with open('C:/caar/College/AI/Pretrained/Unet+KNN/Saved/Unet_Knn.pkl', 'wb') as model_file:pickle.dump(KNN_model, model_file)


#Send test data through same feature extractor process 
X_test_feature = model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)


#Now predict using the trained RF model. 
prediction_KNN = KNN_model.predict(X_test_features) #Inverse le transform to get original label back. 
prediction_KNN = le.inverse_transform(prediction_KNN)

#Print overall accuracy 
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_KNN))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_KNN) #print(cm)
sns.heatmap(cm,annot=True)



 


#Check results on a few select images 
n=np.random.randint(0, x_test.shape[0]) 
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=model.predict(input_img)

input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_KNN = KNN_model.predict(input_img_features)[0]
prediction_KNN=le.inverse_transform({prediction_KNN})
 
#Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_KNN) 
print("The actual label for this image is: ", test_labels[n])


