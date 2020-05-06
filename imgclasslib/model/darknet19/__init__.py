import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D, LeakyReLU
import pickle
import matplotlib.pyplot as plt
from imgclasslib.model.darknet53.block import *

def create_darknet19(IMG_SIZE,num_categories=4):
    inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Block2(inputs,3,1,32)
    x = MaxPooling2D(pool_size=2,strides=2)(x)
    x = Block2(x,3,1,64)
    x = MaxPooling2D(pool_size=2,strides=2)(x)
    x = Block1(x,[3,1],[1,1],[128,64])
    x = Block2(x,3,1,128)
    x = MaxPooling2D(pool_size=2,strides=2)(x)
    x = Block1(x,[3,1],[1,1],[256,128])
    x = Block2(x,3,1,256)   
    x = MaxPooling2D(pool_size=2,strides=2)(x)
    x = Block1(x,[3,1],[1,1],[512,256])
    x = Block1(x,[3,1],[1,1],[512,256])
    x = Block2(x,3,1,512) 
    x = MaxPooling2D(pool_size=2,strides=2)(x)
    x = Block1(x,[3,1],[1,1],[1024,512])
    x = Block1(x,[3,1],[1,1],[1024,512])
    x = Block2(x,3,1,1024)
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(num_categories,activation='softmax')(x)
    model = Model(inputs,outputs)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model