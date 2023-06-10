# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import scipy.optimize as opt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation

#read dataset and drop duplicates
path ='C:\\Users\\Lenovo\\Desktop\\Selected-1 Project\\Datasets\\heart.csv'
data =pd.read_csv(path)
data=data.drop_duplicates()

#make function of h(theta)
plt.figure(figsize=(7, 7))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#make cost function
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

# add a ones column 
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(14)

#cost before optimization
thiscost = cost(theta, X, y)
print('\ncost = ' , thiscost)

#optimize theta
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    return grad



#cost after optimization
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
costafteroptimize = cost(result[0], X, y)
print('\ncost after optimize = ' , costafteroptimize)

#test model
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
pred = [1 if((a==1 and b==1) or 
                (a==0 and b==0))else 0
                for(a,b) in zip(predictions,y)]

print('')
#split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
X_train,X_val,y_train,y_val= train_test_split(X_train,y_train,test_size=0.25,random_state=1)
print("number of X_train" ,X_train.shape)
print("number of y_train" ,y_train.shape)
print("number of X_test" ,X_test.shape)
print("number of y_test" ,y_test.shape)
print("number of X_val",X_val.shape)
print("number of y_val",y_val.shape)


logistic_regression= LogisticRegression(C=1.0) 
logistic_regression.fit(X_train,y_train) 
y_pred=logistic_regression.predict(X_test)

print('')

#show aacuracy
accuracy =(sum(map(int ,pred)) /len(pred))
print ('Accuracy of testing set (model) = {0}%\n'.format(accuracy*100))

#ROC Curve
plot_roc_curve(logistic_regression, X_test, y_test)

#Confusion matrix
cm=confusion_matrix(y_test, y_pred)
print('confusion matrix is \n' ,cm)
binary1 = np.array(cm)
fig, ax = plot_confusion_matrix(conf_mat=binary1)
plt.show()









