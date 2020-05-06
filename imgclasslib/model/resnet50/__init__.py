import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt
from imgclasslib.model.resnet50.block import *

def create_resnet50(IMG_SIZE,num_categories=4):
    inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Conv2D(kernel_size=7,strides=2, filters=64,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
    #resnet block 2a
    x = Block1(x,[(1,3,1),1],[(1,1,1),1],[(64,64,256),256])
    #resnet block 2b
    x = Block2(x,[1,3,1],[1,1,1],[64,64,256])
    #resnet block 2c
    x = Block2(x,[1,3,1],[1,1,1],[64,64,256])
    #resnet block 3a
    x = Block1(x,[(1,3,1),1],[(2,1,1),2],[(128,128,512),512])
    #resnet block 3b
    x = Block2(x,[1,3,1],[1,1,1],[128,128,512])
    #resnet block 3c
    x = Block2(x,[1,3,1],[1,1,1],[128,128,512])
    #resnet block 3d
    x = Block2(x,[1,3,1],[1,1,1],[128,128,512])
    #resnet block 4a
    x = Block1(x,[(1,3,1),1],[(2,1,1),2],[(256,256,1024),1024])
    #resnet block 4b
    x = Block2(x,[1,3,1],[1,1,1],[256,256,1024])
    #resnet block 4c
    x = Block2(x,[1,3,1],[1,1,1],[256,256,1024])
    #resnet block 4d
    x = Block2(x,[1,3,1],[1,1,1],[256,256,1024])
    #resnet blcok 4e
    x = Block2(x,[1,3,1],[1,1,1],[256,256,1024])
    #resnet block 4f
    x = Block2(x,[1,3,1],[1,1,1],[256,256,1024])
    #resnet block 5a
    x = Block1(x,[(1,3,1),1],[(2,1,1),2],[(512,512,2048),2048])
    #resnet block 5b
    x = Block2(x,[1,3,1],[1,1,1],[512,512,2048])
    #resnet block 5c
    x = Block2(x,[1,3,1],[1,1,1],[512,512,2048])
    #output
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(num_categories,activation='softmax')(x)
    model = Model(inputs,outputs)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model