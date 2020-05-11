import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt
from imgclasslib.model.squeezenet.squeeze import Squeeze

def create_squeezenet(IMG_SIZE,num_categories=4):
    inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Conv2D(kernel_size=3,strides=1,filters=64,padding='same',activation='relu')(inputs)
    x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
    #squeeze 1
    x = Squeeze(x,[(1,1,3),(1,1,3)],[(1,1,1),(1,1,1)],[(16,64,64),(16,64,64)])
    x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
    #squeeze 2
    x = Squeeze(x,[(1,1,3),(1,1,3)],[(1,1,1),(1,1,1)],[(32,128,128),(32,128,128)])
    x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)    
    #squeeze 3
    x = Squeeze(x,[(1,1,3),(1,1,3)],[(1,1,1),(1,1,1)],[(32,128,128),(32,128,128)])
    x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
    #squeeze 4
    x = Squeeze(x,[(1,1,3),(1,1,3)],[(1,1,1),(1,1,1)],[(48,192,192),(48,192,192)])
    #squeeze 5
    x = Squeeze(x,[(1,1,3),(1,1,3)],[(1,1,1),(1,1,1)],[(64,256,256),(64,256,256)])
    x = Dropout(0.5)(x)
    x = Conv2D(kernel_size=1,strides=1,filters=1000,padding='same',activation='relu')(x)
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
