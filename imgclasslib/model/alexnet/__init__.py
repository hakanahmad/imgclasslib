import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt

def create_alexnet(IMG_SIZE,num_categories=4):
    inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Conv2D(kernel_size=(11,11),strides=(4,4),filters=48,padding='same')(inputs)
    x = ReLU()(x)
    #first division
    #x1
    x1 = Conv2D(kernel_size=(5,5),strides=(1,1),filters=48,padding='same')(x)
    x1 = ReLU()(x1)
    x1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x1)
    x1 = Conv2D(kernel_size=(3,3),strides=(1,1),filters=128,padding='same')(x1)
    x1 = ReLU()(x1)
    #x2
    x2 = Conv2D(kernel_size=(5,5),strides=(1,1),filters=48,padding='same')(x)
    x2 = ReLU()(x2)
    x2 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x2)
    x2 = Conv2D(kernel_size=(3,3),strides=(1,1),filters=48,padding='same')(x2)
    x2 = ReLU()(x2)
    #concat
    x = Concatenate()([x1,x2])
    x = MaxPooling2D(pool_size=(2,2),padding='same')(x)
    #second divison
    #x1
    x1 = Conv2D(kernel_size=(3,3),strides=(1,1),filters=192,padding='same')(x)
    x1 = ReLU()(x1)
    x1 = Conv2D(kernel_size=(3,3),strides=(1,1),filters=192,padding='same')(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(kernel_size=(3,3),strides=(1,1),filters=128,padding='same')(x1)
    x1 = ReLU()(x1)
    #x2
    x2 = Conv2D(kernel_size=(3,3),strides=(1,1),filters=192,padding='same')(x)
    x2 = ReLU()(x2)
    x2 = Conv2D(kernel_size=(3,3),strides=(1,1),filters=192,padding='same')(x2)
    x2 = ReLU()(x2)
    x2 = Conv2D(kernel_size=(3,3),strides=(1,1),filters=128,padding='same')(x2)
    x2 = ReLU()(x2)
    #concat
    x = Concatenate()([x1,x2])
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    #full connected layer
    x = Flatten()(x)
    x = Dense(4096,activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(4096,activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_categories,activation='softmax')(x)
    model = Model(inputs,outputs)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model