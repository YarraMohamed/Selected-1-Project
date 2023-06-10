# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 17:33:08 2022

@author: Lenovo
"""
#import Libraries
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score,plot_roc_curve
from sklearn.svm import SVC

#Load data 
path ='C:\\Users\\Lenovo\\Desktop\\Selected-1 Project\\Datasets\\heart.csv'
data =pd.read_csv(path)

y = data.target
x = data.drop(columns=["target"])

#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

print("number of X_train" ,x_train.shape)
print("number of y_train" ,y_train.shape)
print("number of X_test" ,x_test.shape)
print("number of y_test" ,y_test.shape)

#Apply SVM
from sklearn import svm
sv = svm.SVC(kernel='linear')
sv.fit(x_train, y_train)
pred_svm = sv.predict(x_test)
print("Accuracy is : {0} % ".format(accuracy_score(y_test, pred_svm)*100))

#Predict
y_pred = sv.predict(x_test)

#Confusion Matrix
cm=confusion_matrix(y_test, y_pred)
print('confusion matrix is \n' ,cm)
binary1 = np.array(cm)
fig, ax = plot_confusion_matrix(conf_mat=binary1)
plt.show()

#ROC Curve
plot_roc_curve(sv, x_test, y_test)





