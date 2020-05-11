import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt

def create_vggnet(IMG_SIZE,num_categories=4):
    inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Conv2D(kernel_size=(3,3),filters=64,padding='same',activation='relu')(inputs)
    x = Conv2D(kernel_size=(3,3),filters=64,padding='same',activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2),padding='same')(x)
    x = Conv2D(kernel_size=(3,3),filters=128,padding='same',activation='relu')(x)
    x = Conv2D(kernel_size=(3,3),filters=128,padding='same',activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2),padding='same')(x)
    x = Conv2D(kernel_size=(3,3),filters=256,padding='same',activation='relu')(x)
    x = Conv2D(kernel_size=(3,3),filters=256,padding='same',activation='relu')(x)
    x = Conv2D(kernel_size=(3,3),filters=256,padding='same',activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2),padding='same')(x)
    x = Conv2D(kernel_size=(3,3),filters=512,padding='same',activation='relu')(x)
    x = Conv2D(kernel_size=(3,3),filters=512,padding='same',activation='relu')(x)
    x = Conv2D(kernel_size=(3,3),filters=512,padding='same',activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2),padding='same')(x)
    x = Conv2D(kernel_size=(3,3),filters=512,padding='same',activation='relu')(x)
    x = Conv2D(kernel_size=(3,3),filters=512,padding='same',activation='relu')(x)
    x = Conv2D(kernel_size=(3,3),filters=512,padding='same',activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2),padding='same')(x)
    x = Flatten()(x)
    x = Dense(4096,activation='relu')(x)
    x = Dense(4096,activation='relu')(x)
    x = Dense(4096,activation='relu')(x)
    outputs = Dense(num_categories,activation='softmax')(x)
    model = Model(inputs,outputs)
    if num_categories == 2:
        loss = 'binary_crossentropy'
    elif num_categories > 2:
        loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
    return model
