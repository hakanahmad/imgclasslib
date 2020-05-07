import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D, LeakyReLU
import pickle
import matplotlib.pyplot as plt
from imgclasslib.model.densenet201.block import *

def create_densenet201(IMG_SIZE,num_categories=4):
    inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = ZeroPadding2D(padding=((3,3),(3,3)))(inputs)
    x = Conv2D(kernel_size=7,strides=2,filters=64)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ZeroPadding2D(padding=((1,1),(1,1)))(x)
    x = MaxPooling2D(pool_size=3,strides=2)(x)
    x = dense_block(x,6)
    x = transition_block(x)
    x = dense_block(x,12)
    x = transition_block(x)
    x = dense_block(x,48)
    x = transition_block(x)
    x = dense_block(x,32)
    x = transition_block(x)
    x = AveragePooling2D(pool_size=7)(x)
    x = Flatten()(x)
    outputs = Dense(num_categories,activation='softmax')(x)
    model = Model(inputs,outputs)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
