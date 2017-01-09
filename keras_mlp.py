#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 22:35:12 2016

@author: dawg
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense , Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K
import numpy as np 
from keras.utils.np_utils import to_categorical

img_rows =28
img_cols = 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], img_rows*img_cols)
    X_test = X_test.reshape(X_test.shape[0],img_rows* img_cols)
    input_shape = (1, img_rows, img_cols)
y_train= to_categorical(y_train, 10)
y_test = to_categorical(y_test,10)

model = Sequential()
model.add(Dense(200, input_dim = 784 ,init = "uniform"))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(200, init = 'uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(75))
model.add(Activation('relu'))
model.add(Dense(10, init= 'uniform'))
model.add(Activation('softmax'))
sgd = SGD(lr = 0.01 , decay = 1e-6 , momentum = 0.9 , nesterov = True)
model.compile(loss = 'categorical_crossentropy',optimizer = sgd,metrics = ['accuracy'])
model.fit(X_train, y_train, nb_epoch = 5, batch_size = 16)
score = model.evaluate(X_test, y_test , batch_size = 16)
