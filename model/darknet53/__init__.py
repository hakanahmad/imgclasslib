import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D, LeakyReLU
import pickle
import matplotlib.pyplot as plt
from imgclasslib.model.darknet53.block import *

def create_darknet53(IMG_SIZE,num_categories=4):
    inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Block1(inputs,[3,3],[1,2],[32,64])
    x1 = Block1(x,[1,3],[1,1],[32,64])
    x = Add()([x,x1])
    x = Block2(x,3,2,128)
    x1 = Block1(x,[1,3],[1,1],[64,128])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[64,128])
    x = Add()([x,x1])
    x = Block2(x,3,2,256)
    x1 = Block1(x,[1,3],[1,1],[128,256])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[128,256])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[128,256])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[128,256])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[128,256])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[128,256])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[128,256])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[128,256])
    x = Add()([x,x1])
    x = Block2(x,3,2,512)
    x1 = Block1(x,[1,3],[1,1],[256,512])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[256,512])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[256,512])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[256,512])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[256,512])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[256,512])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[256,512])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[256,512])
    x = Add()([x,x1])
    x = Block2(x,3,2,1024)
    x1 = Block1(x,[1,3],[1,1],[512,1024])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[512,1024])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[512,1024])
    x = Add()([x,x1])
    x1 = Block1(x,[1,3],[1,1],[512,1024])
    x = Add()([x,x1])    
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(num_categories,activation='softmax')(x)
    model = Model(inputs,outputs)
    if num_categories == 2:
        loss = 'binary_crossentropy'
    elif num_categories > 2:
        loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
    return model
