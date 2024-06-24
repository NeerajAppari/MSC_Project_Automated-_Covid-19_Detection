# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:34:58 2024

@author: appar
"""

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
SIZE = 256


train_images = [] 
train_labels = []
for directory_path in glob.glob("C:/caar/College/AI/Pretrained/Train/*"):
    label = directory_path.split("\\")[-1] 
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        img = cv2.resize(img, (SIZE, SIZE)) 
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
        img = cv2.resize(img, (SIZE, SIZE)) 
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

y_train_reshaped = y_train.reshape(-1, 1,1,1)
y_test_reshaped = y_test.reshape(-1, 1,1,1)

from tensorflow.keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train_reshaped)
y_test_one_hot = to_categorical(y_test_reshaped)



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
 
#outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation=None)(c9)
outputs = tf.keras.layers.Activation('sigmoid')(outputs)


model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train_one_hot, epochs=50, validation_data = (x_test, y_test_one_hot)) 

import pickle

with open('C:/caar/College/AI/UNET2/Saved/unet2.pkl', 'wb') as model_file:pickle.dump(history, model_file)

prediction_NN = model.predict(x_test) 
prediction_NN = np.argmax(prediction_NN, axis=-1) 
prediction_NN = le.inverse_transform(prediction_NN)

from sklearn import metrics
#from sklearn.metrics import classification_report #from sklearn.metrics import average_precision_score
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_NN))

#Confusion Matrix - verify accuracy of each class 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, prediction_NN) 
print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images #n=5 dog park. NN not as good as RF.
n=4 #Select the index of image to be loaded for testing img = x_test[n]
plt.imshow(img)
 
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
prediction = np.argmax(model.predict(input_img)) #argmax to convert categorical back to original
prediction = le.inverse_transform([prediction]) #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction) 
print("The actual label for this image is: ", test_labels[n])
 


