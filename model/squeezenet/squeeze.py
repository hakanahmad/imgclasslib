import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt

def Squeeze(layer,kernel_array,stride_array,filter_array):
    x = Conv2D(kernel_size=kernel_array[0][0],strides=stride_array[0][0],filters=filter_array[0][0],padding='same',activation='relu')(layer)
    x1 = Conv2D(kernel_size=kernel_array[0][1],strides=stride_array[0][1],filters=filter_array[0][1],padding='same',activation='relu')(x)
    x2 = Conv2D(kernel_size=kernel_array[0][2],strides=stride_array[0][2],filters=filter_array[0][2],padding='same',activation='relu')(x)
    x = Concatenate()([x1,x2])
    x = Conv2D(kernel_size=kernel_array[1][0],strides=stride_array[1][0],filters=filter_array[1][0],padding='same',activation='relu')(x)
    x1 = Conv2D(kernel_size=kernel_array[1][1],strides=stride_array[1][1],filters=filter_array[1][1],padding='same',activation='relu')(x)
    x2 = Conv2D(kernel_size=kernel_array[1][2],strides=stride_array[1][2],filters=filter_array[1][2],padding='same',activation='relu')(x)  
    x = Concatenate()([x1,x2])
    return x