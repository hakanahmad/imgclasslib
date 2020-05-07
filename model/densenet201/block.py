import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D, LeakyReLU
import pickle
import matplotlib.pyplot as plt

def conv_block(layer,kernel_array,stride_array,filter_array):
    x1 = BatchNormalization()(layer)
    x1 = ReLU()(x1)
    x1 = Conv2D(kernel_size=kernel_array[0],strides=stride_array[0],filters=filter_array[0])(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(kernel_size=kernel_array[1],strides=stride_array[1],filters=filter_array[1])(x1)
    x = Concatenate(axis=-1)([x,x1])
    return x

def dense_block(layer,blocks):
    for i in range(blocks):
        x = conv_block(layer,[1,3],[1,1],[4*32,32])
    return x

def transition_block(layer):
    x = BatchNormalization()(layer)
    x = ReLU()(x)
    x = Conv2D(kernel_size=1,strides=2,filters=32)(x)
    x = AveragePooling2D()(x)
    return x