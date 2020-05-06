import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt
from imgclasslib.model.googlenet.inception import Inception

def create_googlenet(IMG_SIZE,num_categories=4):
    inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Conv2D(kernel_size=(7,7),strides=2,filters=64,padding='same',activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(kernel_size=(1,1),filters=64,strides=1,padding='same',activation='relu')(x)
    x = Conv2D(kernel_size=(3,3),filters=192,strides=1,padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #inception3a   layer = inception(layer, [ 64,  (96,128), (16,32), 32])
    x = Inception(x,[64,(96,128),(16,32),32])
    #inception3b    layer = inception(layer, [128, (128,192), (32,96), 64]) 
    x = Inception(x,[128,(128,192),(32,96),64]) 
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)  
    
    #inception4a   layer = inception(layer, [192,  (96,208),  (16,48),  64]) 
    x = Inception(x,[192,(96,208),(16,48),64])
    #inception4b and first output   layer = inception(layer, [160, (112,224),  (24,64),  64]) 
    layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
    layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
    layer = Flatten()(layer)
    layer = Dense(units=256, activation='relu')(layer)
    layer = Dropout(0.4)(layer)
    outputs1 = Dense(units=num_categories, activation='softmax')(layer)
    x = Inception(x,[160,(122,224),(24,64),64])   
    #inception4c   layer = inception(layer, [128, (128,256),  (24,64),  64]) 
    x = Inception(x,[128,(128,256),(24,64),64])
    #inception4d and second output   layer = inception(layer, [112, (144,288),  (32,64),  64]) 
    x = Inception(x,[112,(114,228),(32,64),64])
    layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
    layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
    layer = Flatten()(layer)
    layer = Dense(units=256, activation='relu')(layer)
    layer = Dropout(0.4)(layer)
    outputs2 = Dense(units=num_categories, activation='softmax')(layer)   
    #inception4e  layer = inception(layer, [256, (160,320), (32,128), 128])
    x = Inception(x,[256,(160,320),(32,128),128])
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
    
    #inception5a    layer = inception(layer, [256, (160,320), (32,128), 128])
    x = Inception(x,[256,(160,320),(32,128),128])
    #inception5b    layer = inception(layer, [384, (192,384), (48,128), 128])
    x = Inception(x,[384,(192,384),(48,128),128])
    
    #third output
    x = AveragePooling2D(pool_size=(7,7),strides=1)(x)
    x = Flatten()(x)
    x = Dense(256,activation='linear')(x)
    x = Dropout(0.4)(x)
    outputs3 = Dense(num_categories,activation='softmax')(x)
    #outputs = Average()([outputs1,outputs2,outputs3])
    model = Model(inputs,[outputs1,outputs2,outputs3])
    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model