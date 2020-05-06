import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt

def Block1(layer,kernel_size,stride_size,filter_size):
    x1 = Conv2D(kernel_size=kernel_size[0][0],strides=stride_size[0][0],filters=filter_size[0][0],padding='same')(layer)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(kernel_size=kernel_size[0][1],strides=stride_size[0][1],filters=filter_size[0][1],padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(kernel_size=kernel_size[0][2],strides=stride_size[0][2],filters=filter_size[0][2],padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(kernel_size=kernel_size[1],strides=stride_size[1],filters=filter_size[1],padding='same')(layer)
    x2 = BatchNormalization()(x2)
    x = Add()([x1,x2])
    x = ReLU()(x)
    return x
    
def Block2(layer,kernel_size,stride_size,filter_size):
    x1 = Conv2D(kernel_size=kernel_size[0],strides=stride_size[0],filters=filter_size[0],padding='same')(layer)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(kernel_size=kernel_size[1],strides=stride_size[1],filters=filter_size[1],padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(kernel_size=kernel_size[2],strides=stride_size[2],filters=filter_size[2],padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x = Add()([layer,x1])
    x = ReLU()(x)
    return x