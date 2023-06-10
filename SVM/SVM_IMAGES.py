# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 23:10:03 2022

@author: Lenovo
"""

#import Libraries
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix , plot_roc_curve
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import glob
import os
import cv2
import glob

#Load trian set
fruit_images = []
labels = [] 
for fruit_dir_path in glob.glob("C:\\Users\\Lenovo\\Desktop\\Selected-1 Project\\Datasets\\fruits-360_dataset\\Training/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)

label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in labels])


#Load Test set
validation_fruit_images = []
validation_labels = [] 
for fruit_dir_path in glob.glob("C:\\Users\\Lenovo\\Desktop\\Selected-1 Project\\Datasets\\fruits-360_dataset\\Test/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        validation_fruit_images.append(image)
        validation_labels.append(fruit_label)
validation_fruit_images = np.array(validation_fruit_images)
validation_labels = np.array(validation_labels)

label_to_id_dict = {v:i for i,v in enumerate(np.unique(validation_labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])


#Making the train sets and test sets
X_train, X_test = fruit_images, validation_fruit_images
Y_train, Y_test = label_ids, validation_label_ids

#Normalize color values to between 0 and 1
X_train = X_train/255
X_test = X_test/255

X_train= np.array(X_train).reshape(19000,-1)
X_test = np.array(X_test).reshape(6356,-1)

#apply SVM
svc = SVC(kernel='linear',gamma='auto')
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)

print("Accuracy is : {0} % ".format(accuracy_score(Y_test, y_pred)*100))

#Confusion Matrix
cm=confusion_matrix(Y_test, y_pred)
print('confusion matrix is \n' ,cm)
binary1 = np.array(cm)
fig, ax = plot_confusion_matrix(conf_mat=binary1,figsize=(10,10))
plt.show()


from sklearn.metrics import auc,roc_curve
fpr, tpr, thresholds = roc_curve(Y_test,y_pred ,pos_label=1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='SVM')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.show()
