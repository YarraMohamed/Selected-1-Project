# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:20:05 2022

@author: Lenovo
"""
#to produce a fixed accuracy each time
import os
os.environ['PYTHONHASHSEED']= '0'
import numpy as np
np.random.seed(0)
import random as rn
rn.seed(0)
import tensorflow as tf
tf.random.set_seed(0)

#import needed libraries
import numpy as np 
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import glob
import cv2
import os
import numpy as np 
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.models import Model


#Load Training sets
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

#Load Test sets
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

#One Hot Encode the Output
Y_train = keras.utils.to_categorical(Y_train, 60)
Y_test = keras.utils.to_categorical(Y_test, 60)



#bulid the model
model_dense = Sequential()
model_dense.add(Flatten())

model_dense.add(Dense(128, activation='relu'))
model_dense.add(Dense(64, activation='relu'))
model_dense.add(Dense(60, activation='softmax'))

model_dense.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              run_eagerly=True)

history= model_dense.fit(X_train, Y_train,
                          epochs=50,
                          verbose=1,
                          validation_data=(X_test, Y_test))

#Show Accuracy of model
score = model_dense.evaluate(X_test, Y_test, verbose=0)
print('Test loss:',format(score[0]*100))
print('Test accuracy:', format(score[1]*100))

print()

#loss Curve
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#confusion Matrix 
y_pred=model_dense.predict(X_test) 
y_pred=np.argmax(y_pred, axis=1)
Y_test=np.argmax(Y_test, axis=1)
cm = confusion_matrix(Y_test, y_pred)
print('confusion matrix is \n',cm)
binary1 = np.array(cm)
fig, ax = plot_confusion_matrix(conf_mat=binary1 , figsize=(10,10))
plt.show()

# y_prediction = []
# for i in range(0 ,len(y_pred)):
#     classes_x=np.argmax(y_pred[i])
#     y_prediction.append(classes_x)
# y_prediction = np.asarray(y_prediction)
# y_prediction

from sklearn.metrics import auc,roc_curve
fpr, tpr, thresholds = roc_curve(Y_test,y_pred ,pos_label=1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='ANN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.show()


