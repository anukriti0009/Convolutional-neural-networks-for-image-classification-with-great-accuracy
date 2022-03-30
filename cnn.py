# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 23:33:45 2022

@author: anukr
""" #cnns
import keras
import numpy as np
import pandas as pd
import matplotlib as plt
from keras.models import Sequential #to make our model sequential
from keras.layers import Convolution2D #to add convolution layers
from keras.layers import MaxPooling2D #pooling step 
from keras.layers import Flatten #for flattening to convert polled maps into large feature network that becomes input
from keras.layers import Dense

#initialising cnn
classifier = keras.models.Sequential([
    keras.layers.Conv2D(32,3,3,activation = 'relu',input_shape = [64,64,3]),
    keras.layers.MaxPooling2D(2,strides=2),
    keras.layers.Flatten(),
    keras.layers.Conv2D(32,3,3,activation = 'relu'),
    keras.layers.MaxPooling2D(2,strides=2),
    keras.layers.Dense(128,activation = 'relu'),
    keras.layers.Dense(1,activation = 'sigmoid')])


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=8000,
        validation_data= test_set,
        validation_steps=2000)


















