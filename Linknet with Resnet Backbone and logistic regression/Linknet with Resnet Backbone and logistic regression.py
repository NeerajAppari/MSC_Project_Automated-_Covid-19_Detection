# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:18:12 2024

@author: appar
"""

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
from tensorflow.keras.applications.vgg16 import VGG16 
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



# Normalize pixel values to between 0 and 1 x_train, 
x_test = x_train / 255.0, x_test / 255.0

from tensorflow.keras.utils import to_categorical 
y_train_one_hot = to_categorical(y_train) 
y_test_one_hot = to_categorical(y_test)


from segmentation_models import Linknet
from segmentation_models import get_preprocessing 
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score


BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)



# define model
model = Linknet(BACKBONE, input_shape=(224, 224, 3), encoder_weights=None)
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])


#Now, let us use features from convolutional network for RF 
feature_extractor=model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)


X_for_Logreg = features #This is our X input to RF 
from sklearn.linear_model import LogisticRegression
Logreg_model= LogisticRegression(random_state=1)
 



# #RANDOM FOREST
# from sklearn.ensemble import RandomForestClassifier
# RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)


# Train the model on training data
Logreg_model.fit(X_for_Logreg, y_train) #For sklearn no one hot encoding


#Send test data through same feature extractor process 
X_test_feature = model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)


#Now predict using the trained RF model. 
prediction_Logreg = Logreg_model.predict(X_test_features) #Inverse le transform to get original label back.
prediction_Logreg = le.inverse_transform(prediction_Logreg) #Print overall accuracy
 
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_Logreg))

#Confusion Matrix - verify accuracy of each class 
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_Logreg) #print(cm)
sns.heatmap(cm, annot=True)




#Check results on a few select images
 
n=np.random.randint(0, x_test.shape[0]) 
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_Logreg = Logreg_model.predict(input_img_features)[0]
prediction_Logreg = le.inverse_transform([prediction_Logreg]) #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_Logreg)
print("The actual label for this image is: ", test_labels[n])
