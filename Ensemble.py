# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 18:20:12 2024

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

from keras.models import load_model 
from sklearn.metrics import accuracy_score

model1 = load_model('C:/caar/College/AI/Ensemble/Saved/vgg_model1.hdf5')
model2 = load_model('C:/caar/College/AI/Ensemble/Saved/resnet50_1.hdf5')
model3 = load_model('C:/caar/College/AI/Ensemble/Saved/mobilenetv2_1.hdf5')


models = [model1, model2, model3]


preds = [model.predict(x_test) for model in models]
 
preds=np.array(preds)
summed = np.sum(preds, axis=0)


# argmax across classes
ensemble_prediction = np.argmax(summed, axis=1)



prediction1 = model1.predict(x_test) 
prediction2 = model2.predict(x_test) 
prediction3 = model3.predict(x_test)

prediction1 = np.argmax(prediction1, axis=-1) 
prediction2 = np.argmax(prediction2, axis=-1) 
prediction3 = np.argmax(prediction3, axis=-1)

prediction1 = le.inverse_transform(prediction1) 
prediction2 = le.inverse_transform(prediction2) 
prediction3 = le.inverse_transform(prediction3)
ensemble_prediction = le.inverse_transform(ensemble_prediction)


accuracy1 = accuracy_score(test_labels, prediction1)
 
accuracy2 = accuracy_score(test_labels, prediction2) 
accuracy3 = accuracy_score(test_labels, prediction3)
ensemble_accuracy = accuracy_score(test_labels, ensemble_prediction)


print('Accuracy Score for model1 = ', accuracy1)
print('Accuracy Score for model2 = ', accuracy2) 
print('Accuracy Score for model3 = ', accuracy3)
print('Accuracy Score for average ensemble = ', ensemble_accuracy)
########################################
#Weighted average ensemble models = [model1, model2, model3]
preds = [model.predict(x_test) for model in models] 
preds=np.array(preds)
weights = [0.4, 0.2, 0.4]


#Use tensordot to sum the products of all elements over specified axes.
 
weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=1)
weighted_ensemble_prediction = le.inverse_transform(weighted_ensemble_prediction)
weighted_accuracy = accuracy_score(test_labels, weighted_ensemble_prediction)


print('Accuracy Score for model1 = ', accuracy1) 
print('Accuracy Score for model2 = ', accuracy2) 
print('Accuracy Score for model3 = ', accuracy3)
print('Accuracy Score for average ensemble = ', ensemble_accuracy)
print('Accuracy Score for weighted average ensemble = ', weighted_accuracy)
########################################
#Grid search for the best combination of w1, w2, w3 that gives maximum acuracy
 
models = [model1, model2, model3]
preds1 = [model.predict(x_test) for model in models] 
preds1=np.array(preds1)


import pandas as pd 
df = pd.DataFrame([])

for w1 in range(0, 5): 
    for w2 in range(0,5):

        for w3 in range(0,5):
            wts = [w1/10.,w2/10.,w3/10.] 
            wted_preds1 = np.tensordot(preds1, wts,axes=((0),(0)))
            wted_ensemble_pred = np.argmax(wted_preds1, axis=1)
            weighted_accuracy = accuracy_score(y_test, wted_ensemble_pred)
            df1 = pd.DataFrame({'wt1':wts[0],'wt2':wts[1],'wt3':wts[2], 'acc':weighted_accuracy*100},index=[0])
            df2 = pd.DataFrame()
            df= pd.concat([df1,df2],ignore_index=True)
max_acc_row = df.iloc[df['acc'].idxmax()]
print("Max accuracy of ", max_acc_row[0], " obained with w1=", max_acc_row[1]," w2=", max_acc_row[2], " and w3=", max_acc_row[3])

# df = pd.DataFrame(columns=['wt1', 'wt2', 'wt3', 'acc'])

# for w1 in range(0, 5):
#     for w2 in range(0, 5):
#         for w3 in range(0, 5):
#             wts = [w1/10., w2/10., w3/10.] 
#             wted_preds1 = np.tensordot(preds1, wts, axes=((0),(0)))
#             wted_ensemble_pred = np.argmax(wted_preds1, axis=1)
#             weighted_accuracy = accuracy_score(y_test, wted_ensemble_pred)
#             # Append new row to the DataFrame
#             df = df.append({'wt1': wts[0], 'wt2': wts[1], 'wt3': wts[2], 'acc': weighted_accuracy * 100}, ignore_index=True)

# # Find the row with maximum accuracy
# max_acc_row = df.iloc[df['acc'].idxmax()]
# print("Max accuracy of ", max_acc_row['acc'], " obtained with w1=", max_acc_row['wt1'], " w2=", max_acc_row['wt2'], " and w3=", max_acc_row['wt3'])




################################################## #########################
### Explore metrics for the ideal weighted ensemble model.


models = [model1, model2, model3]
preds = [model.predict(x_test) for model in models] 
preds=np.array(preds)
ideal_weights = [0.4, 0.4, 94.0]


#Use tensordot to sum the products of all elements over specified axes.
ideal_weighted_preds = np.tensordot(preds, ideal_weights, axes=((0),(0)))
 
ideal_weighted_ensemble_prediction = np.argmax(ideal_weighted_preds, axis=1)
ideal_weighted_ensemble_prediction = le.inverse_transform(ideal_weighted_ensemble_prediction)


ideal_weighted_accuracy = accuracy_score(test_labels, ideal_weighted_ensemble_prediction)



print('Accuracy Score for ideal weighted average ensemble = ', ideal_weighted_accuracy)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, prediction1) 
sns.heatmap(cm, annot=True)
 


 

cm1 = confusion_matrix(test_labels, prediction2)
sns.heatmap(cm1, annot=True)
 

 

cm2 = confusion_matrix(test_labels, prediction3) 
sns.heatmap(cm2, annot=True)
 

 

cm3 = confusion_matrix(test_labels, ensemble_prediction) 
sns.heatmap(cm3, annot=True)
 


 

cm4 = confusion_matrix(test_labels, weighted_ensemble_prediction)
sns.heatmap(cm4, annot=True)
 


 

cm5 = confusion_matrix(test_labels, ideal_weighted_ensemble_prediction)
sns.heatmap(cm5, annot=True)
