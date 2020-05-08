import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt

def Block1(layer,groups,kernel_size,stride_size,filter_size,relu=True):
    if relu==True:
        x1 = ReLU()(layer)
    else:
        x1 = layer
    a0 = []
    for i in range(groups[0]):
        a0.append(Conv2D(kernel_size=kernel_size[0][0],strides=stride_size[0][0],filters=filter_size[0][0],padding='same')(x1))
    x1 = Add()(a0)
    x1 = Conv2D(kernel_size=kernel_size[0][1],strides=stride_size[0][1],filters=filter_size[0][1],padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    a1 = []
    for i in range(groups[1]):
        a1.append(Conv2D(kernel_size=kernel_size[0][2],strides=stride_size[0][2],filters=filter_size[0][2],padding='same')(x1))
    x1 = Add()(a1)
    x1 = Conv2D(kernel_size=kernel_size[0][3],strides=stride_size[0][3],filters=filter_size[0][3],padding='same')(x1)
    x1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x1)
    x2 = Conv2D(kernel_size=kernel_size[1],strides=stride_size[1],filters=filter_size[1],padding='same')(layer)
    x2 = BatchNormalization()(x2)
    x = Add()([x1,x2])
    return x
    
def Block2(layer,groups,kernel_size,stride_size,filter_size):
    x1 = ReLU()(layer)
    a0 = []
    for i in range(groups[0]):
        a0.append(Conv2D(kernel_size=kernel_size[0],strides=stride_size[0],filters=filter_size[0],padding='same')(x1))
    x1 = Add()(a0)
    x1 = Conv2D(kernel_size=kernel_size[1],strides=stride_size[1],filters=filter_size[1],padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    a1 = []
    for i in range(groups[1]):
        a1.append(Conv2D(kernel_size=kernel_size[2],strides=stride_size[2],filters=filter_size[2],padding='same')(x1))
    x1 = Add()(a1)
    x1 = Conv2D(kernel_size=kernel_size[3],strides=stride_size[3],filters=filter_size[3],padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    a2 = []
    for i in range(groups[2]):
        a2.append(Conv2D(kernel_size=kernel_size[4],strides=stride_size[4],filters=filter_size[4],padding='same')(x1))
    x1 = Add()(a2)
    x1 = Conv2D(kernel_size=kernel_size[5],strides=stride_size[5],filters=filter_size[5],padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x = Add()([layer,x1])
    return x
    
def Block3(layer,groups,kernel_size,stride_size,filter_size):
    a0 = []
    for i in range(groups[0]):
        a0.append(Conv2D(kernel_size=kernel_size[0],strides=stride_size[0],filters=filter_size[0],padding='same')(layer))
    x1 = Add()(a0)
    x1 = Conv2D(kernel_size=kernel_size[1],strides=stride_size[1],filters=filter_size[1],padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    a1 = []
    for i in range(groups[1]):
        a1.append(Conv2D(kernel_size=kernel_size[2],strides=stride_size[2],filters=filter_size[2],padding='same')(x1))
    x1 = Add()(a1)
    x1 = Conv2D(kernel_size=kernel_size[3],strides=stride_size[3],filters=filter_size[3],padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    return x1