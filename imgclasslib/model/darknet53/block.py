import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D, LeakyReLU
import pickle
import matplotlib.pyplot as plt

def Block1(layer,kernel_array,stride_array,filter_array):
    x = Conv2D(kernel_size=kernel_array[0],strides=stride_array[0],filters=filter_array[0],padding='same')(layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(kernel_size=kernel_array[1],strides=stride_array[1],filters=filter_array[1],padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    return x

def Block2(layer,kernel,stride,filters):
    x = Conv2D(kernel_size=kernel,strides=stride,filters=filters,padding='same')(layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    return x