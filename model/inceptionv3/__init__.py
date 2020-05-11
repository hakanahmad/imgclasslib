import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt
from imgclasslib.model.inceptionv3.inception import *

def create_inceptionv3(IMG_SIZE,num_categories=4):
    inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Conv2D(kernel_size=3,strides=2,filters=32,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(kernel_size=3,strides=1,filters=32,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(kernel_size=3,strides=1,filters=64,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
    x = Conv2D(kernel_size=1,strides=1,filters=80,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(kernel_size=3,strides=1,filters=192,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
    
    #inception1
    x = Inception1(x,[1,(1,5),(1,3,3),1],[1,(1,1),(1,1,1),1],[64,(48,64),(64,96,96),32])
    #inception2 
    x = Inception1(x,[1,(1,5),(1,3,3),1],[1,(1,1),(1,1,1),1],[64,(48,64),(64,96,96),64])    
    #inception3
    x = Inception1(x,[1,(1,5),(1,3,3),1],[1,(1,1),(1,1,1),1],[64,(48,64),(64,96,96),64])    
    #inception4
    x = Inception2(x,[3,(1,3,3)],[2,(1,1,2)],[384,(64,96,96)])
    
    #inception5
    x = Inception3(x,[1,(1,(1,7),(7,1)),(1,(7,1),(1,7),(7,1),(1,7)),1],[1,(1,1,1),(1,1,1,1,1),1],[192,(128,128,192),(128,128,128,128,192),192])
    #inception6
    x = Inception3(x,[1,(1,(1,7),(7,1)),(1,(7,1),(1,7),(7,1),(1,7)),1],[1,(1,1,1),(1,1,1,1,1),1],[192,(160,160,192),(160,160,160,160,192),192])
    #inception7
    x = Inception3(x,[1,(1,(1,7),(7,1)),(1,(7,1),(1,7),(7,1),(1,7)),1],[1,(1,1,1),(1,1,1,1,1),1],[192,(160,160,192),(160,160,160,160,192),192])    
    #inception8
    x = Inception3(x,[1,(1,(1,7),(7,1)),(1,(7,1),(1,7),(7,1),(1,7)),1],[1,(1,1,1),(1,1,1,1,1),1],[192,(192,192,192),(192,192,192,192,192),192])
    #inception9
    x = Inception4(x,[(1,3),(1,(1,7),(7,1),3)],[(1,2),(1,1,1,2)],[(192,320),(192,192,192,192)])
    #inception10
    x = Inception5(x,[1,(1,(1,3),(3,1)),(1,3,(1,3),(3,1)),1],[1,(1,1,1),(1,1,1,1),1],[320,(384,384,384),(448,384,384,384),192])
    #inception11
    x = Inception5(x,[1,(1,(1,3),(3,1)),(1,3,(1,3),(3,1)),1],[1,(1,1,1),(1,1,1,1),1],[320,(384,384,384),(448,384,384,384),192])
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
