import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt

def create_lenet(IMG_SIZE,num_categories=4):
    inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Conv2D(kernel_size=(5,5),strides=(1,1),filters=6,padding='same',activation='tanh')(inputs)
    x = AveragePooling2D(pool_size=(2,2),strides=(1,1),padding='same')(x)
    x = Conv2D(kernel_size=(5,5),strides=(1,1),filters=16,padding='same',activation='tanh')(x)
    x = AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)
    x = Conv2D(kernel_size=(5,5),strides=(1,1),filters=120,padding='same',activation='tanh')(x)
    x = Flatten()(x)
    x = Dense(84,activation='tanh')(x)
    outputs = Dense(num_categories,activation='softmax')(x)
    model = Model(inputs,outputs)
    model.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model