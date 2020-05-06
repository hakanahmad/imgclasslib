import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt

def Inception(layer,filter_array):
    x4 = Conv2D(kernel_size=(1,1),strides=1,filters=filter_array[0],padding='same',activation='relu')(layer)
    
    x31 = Conv2D(kernel_size=(1,1),strides=1,filters=filter_array[1][0],padding='same',activation='relu')(layer)
    x3 = Conv2D(kernel_size=(3,3),strides=1,filters=filter_array[1][1],padding='same',activation='relu')(x31)
    
    x21 = Conv2D(kernel_size=(1,1),strides=1,filters=filter_array[2][0],padding='same',activation='relu')(layer)
    x2 = Conv2D(kernel_size=(5,5),strides=1,filters=filter_array[2][1],padding='same',activation='relu')(x21)    
    
    x11 = MaxPooling2D(pool_size=(3,3),strides=1,padding='same')(layer)
    x1 = Conv2D(kernel_size=(1,1),strides=1,filters=filter_array[3],padding='same',activation='relu')(x11)    
    
    x = Concatenate(axis=-1)([x4,x3,x2,x1])
    return x