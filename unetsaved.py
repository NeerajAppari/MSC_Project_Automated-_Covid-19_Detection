# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:59:04 2023

@author: appar
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nibabel as nib #for loading nii files
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

data = pd.read_csv('C:/caar/College/AI/Unet/input/covid19-ct-scans/metadata1.csv')
data.head()

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)

# Read sample
sample_ct   = read_nii(data.loc[0,'ct_scan'])
sample_lung = read_nii(data.loc[0,'lung_mask'])
sample_infe = read_nii(data.loc[0,'infection_mask'])
sample_all  = read_nii(data.loc[0,'lung_and_infection_mask'])

fig = plt.figure(figsize = (18,15))
plt.subplot(1,4,1)
plt.imshow(sample_ct[..., 150], cmap = 'bone')
plt.title('Original Image')

plt.subplot(1,4,2)
plt.imshow(sample_ct[..., 150], cmap = 'bone')
plt.imshow(sample_lung[..., 150],alpha = 0.5, cmap = 'nipy_spectral')
plt.title('Lung Mask')

plt.subplot(1,4,3)
plt.imshow(sample_ct[..., 150], cmap = 'bone')
plt.imshow(sample_infe[..., 150], alpha = 0.5, cmap = 'nipy_spectral')
plt.title('Infection Mask')

plt.subplot(1,4,4)
plt.imshow(sample_ct[..., 150], cmap = 'bone')
plt.imshow(sample_all[..., 150], alpha = 0.5, cmap = 'nipy_spectral')
plt.title('Lung and Infection Mask')

lungs = []
infections = []
img_size = 128

for i in range(len(data)):
    ct = read_nii(data['ct_scan'][i])
    infect = read_nii(data['infection_mask'][i])
    
    for ii in range(ct.shape[0]):
        lung_img = cv2.resize(ct[ii], dsize = (img_size, img_size),interpolation = cv2.INTER_AREA).astype('uint8')
        infec_img = cv2.resize(infect[ii],dsize=(img_size, img_size),interpolation = cv2.INTER_AREA).astype('uint8')
        lungs.append(lung_img[..., np.newaxis])
        infections.append(infec_img[..., np.newaxis])
        
lungs = np.array(lungs)
infections = np.array(infections)

from sklearn.model_selection import train_test_split
lung_train, lung_test, infect_train, infect_test = train_test_split(lungs, infections, test_size = 0.1)

from keras.models import load_model
from sklearn.metrics import accuracy_score
model1 = load_model('C:/caar/College/AI/Unet/saved/unet1.hdf5')

plt.plot(model1.model1['accuracy'])
plt.plot(model1.model1['val_accuracy'])
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()

plt.plot(model1.model1['loss'])
plt.plot(model1.model1['val_loss'])
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()


predicted = model1.predict(lung_test)
fig = plt.figure(figsize = (18,15))


for i in range(len(lung_test)):
    plt.subplot(1,3,1)
    plt.imshow(lung_test[i][...,0], cmap = 'bone')
    plt.title('original lung')

    plt.subplot(1,3,2)
    plt.imshow(lung_test[i][...,0], cmap = 'bone')
    plt.imshow(infect_test[i][...,0],alpha = 0.5, cmap = "nipy_spectral")
    plt.title('original infection mask')

    plt.subplot(1,3,3)
    plt.imshow(lung_test[i][...,0], cmap = 'bone')
    plt.imshow(predicted[i][...,0],alpha = 0.5,cmap = "nipy_spectral")
    plt.title('predicted infection mask')
    
# accuracy1 = accuracy_score(infect_test, predicted)
# print('Accuracy Score for model1 = ', accuracy1)

# #prediction_NN = cnn_model.predict(x_test)
# predicted = np.argmax(predicted, axis=-1)
# #predicted = le.inverse_transform(predicted)
# #Confusion Matrix - verify accuracy of each class
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(infect_test, predicted)
# print(cm)

# import seaborn as sns
# sns.heatmap(cm, annot=True)
