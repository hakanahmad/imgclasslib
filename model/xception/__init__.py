import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt
from imgclasslib.model.xception.block import *

def create_xception(IMG_SIZE,num_categories=4):
    inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Conv2D(kernel_size=3,strides=2, filters=32,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(kernel_size=3,strides=1, filters=64,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #block 1
    x = Block1(x,[64,128],[(3,1,3,1),1],[(1,1,1,1),2],[(1,128,1,128),128],relu=False)
    #block 2
    x = Block1(x,[128,256],[(3,1,3,1),1],[(1,1,1,1),2],[(1,256,1,256),256])
    #block 3
    x = Block1(x,[256,728],[(3,1,3,1),1],[(1,1,1,1),2],[(1,728,1,728),728])
    #block 4
    x = Block2(x,[728,728,728],[3,1,3,1,3,1],[1,1,1,1,1,1],[1,728,1,728,1,728])
    #block 5
    x = Block2(x,[728,728,728],[3,1,3,1,3,1],[1,1,1,1,1,1],[1,728,1,728,1,728])
    #block 6
    x = Block2(x,[728,728,728],[3,1,3,1,3,1],[1,1,1,1,1,1],[1,728,1,728,1,728])
    #block 7
    x = Block2(x,[728,728,728],[3,1,3,1,3,1],[1,1,1,1,1,1],[1,728,1,728,1,728])
    #block 8
    x = Block2(x,[728,728,728],[3,1,3,1,3,1],[1,1,1,1,1,1],[1,728,1,728,1,728])
    #block 9
    x = Block2(x,[728,728,728],[3,1,3,1,3,1],[1,1,1,1,1,1],[1,728,1,728,1,728])
    #block 10
    x = Block2(x,[728,728,728],[3,1,3,1,3,1],[1,1,1,1,1,1],[1,728,1,728,1,728])
    #block 11
    x = Block2(x,[728,728,728],[3,1,3,1,3,1],[1,1,1,1,1,1],[1,728,1,728,1,728])
    #block 12
    x = Block1(x,[728,728],[(3,1,3,1),1],[(1,1,1,1),2],[(1,728,1,1024),1024])
    x = Block3(x,[1024,1536],[3,1,3,1],[1,1,1,1],[1,1536,1,2048])
    #output
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(num_categories,activation='softmax')(x)
    model = Model(inputs,outputs)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model