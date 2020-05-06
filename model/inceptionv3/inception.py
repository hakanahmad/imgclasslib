import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, Average, ZeroPadding2D
import pickle
import matplotlib.pyplot as plt

def Inception1(layer,kernel_array,stride_array,filter_array):
    x1 = Conv2D(kernel_size=kernel_array[0],strides=stride_array[0],filters=filter_array[0],padding='same')(layer)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    
    x2 = Conv2D(kernel_size=kernel_array[1][0],strides=stride_array[1][0],filters=filter_array[1][0],padding='same')(layer)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Conv2D(kernel_size=kernel_array[1][1],strides=stride_array[1][1],filters=filter_array[1][1],padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    
    x3 = Conv2D(kernel_size=kernel_array[2][0],strides=stride_array[2][0],filters=filter_array[2][0],padding='same')(layer)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Conv2D(kernel_size=kernel_array[2][1],strides=stride_array[2][1],filters=filter_array[2][1],padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Conv2D(kernel_size=kernel_array[2][2],strides=stride_array[2][2],filters=filter_array[2][2],padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    
    x4 = MaxPooling2D(pool_size=3,strides=1,padding='same')(layer)
    x4 = Conv2D(kernel_size=kernel_array[3],strides=stride_array[3],filters=filter_array[3],padding='same')(x4)
    x4 = BatchNormalization()(x4)
    x4 = ReLU()(x4)    
    x = Concatenate(axis=-1)([x1,x2,x3,x4])
    return x

def Inception2(layer,kernel_array,stride_array,filter_array):
    x1 = Conv2D(kernel_size=kernel_array[0],strides=stride_array[0],filters=filter_array[0],padding='same')(layer)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    
    x2 = Conv2D(kernel_size=kernel_array[1][0],strides=stride_array[1][0],filters=filter_array[1][0],padding='same')(layer)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Conv2D(kernel_size=kernel_array[1][1],strides=stride_array[1][1],filters=filter_array[1][1],padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)   
    x2 = Conv2D(kernel_size=kernel_array[1][2],strides=stride_array[1][2],filters=filter_array[1][2],padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)       
    
    x3 = MaxPooling2D(pool_size=3,strides=2,padding='same')(layer)
    x = Concatenate(axis=-1)([x1,x2,x3])
    return x

def Inception3(layer,kernel_array,stride_array,filter_array):
    x1 = Conv2D(kernel_size=kernel_array[0],strides=stride_array[0],filters=filter_array[0],padding='same')(layer)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    
    x2 = Conv2D(kernel_size=kernel_array[1][0],strides=stride_array[1][0],filters=filter_array[1][0],padding='same')(layer)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Conv2D(kernel_size=kernel_array[1][1],strides=stride_array[1][1],filters=filter_array[1][1],padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Conv2D(kernel_size=kernel_array[1][2],strides=stride_array[1][2],filters=filter_array[1][2],padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    
    x3 = Conv2D(kernel_size=kernel_array[2][0],strides=stride_array[2][0],filters=filter_array[2][0],padding='same')(layer)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Conv2D(kernel_size=kernel_array[2][1],strides=stride_array[2][1],filters=filter_array[2][1],padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Conv2D(kernel_size=kernel_array[2][2],strides=stride_array[2][2],filters=filter_array[2][2],padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Conv2D(kernel_size=kernel_array[2][3],strides=stride_array[2][3],filters=filter_array[2][3],padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Conv2D(kernel_size=kernel_array[2][4],strides=stride_array[2][4],filters=filter_array[2][4],padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    
    x4 = MaxPooling2D(pool_size=3,strides=1,padding='same')(layer)
    x4 = Conv2D(kernel_size=kernel_array[3],strides=stride_array[3],filters=filter_array[3],padding='same')(x4)
    x4 = BatchNormalization()(x4)
    x4 = ReLU()(x4)    
    x = Concatenate(axis=-1)([x1,x2,x3,x4])  
    return x

def Inception4(layer,kernel_array,stride_array,filter_array):
    x1 = Conv2D(kernel_size=kernel_array[0][0],strides=stride_array[0][0],filters=filter_array[0][0],padding='same')(layer)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(kernel_size=kernel_array[0][1],strides=stride_array[0][1],filters=filter_array[0][1],padding='same')(layer)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    
    x2 = Conv2D(kernel_size=kernel_array[1][0],strides=stride_array[1][0],filters=filter_array[1][0],padding='same')(layer)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Conv2D(kernel_size=kernel_array[1][1],strides=stride_array[1][1],filters=filter_array[1][1],padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)   
    x2 = Conv2D(kernel_size=kernel_array[1][2],strides=stride_array[1][2],filters=filter_array[1][2],padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)     
    x2 = Conv2D(kernel_size=kernel_array[1][3],strides=stride_array[1][3],filters=filter_array[1][3],padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)   
    
    x3 = MaxPooling2D(pool_size=3,strides=2,padding='same')(layer)
    x = Concatenate(axis=-1)([x1,x2,x3])
    return x

def Inception5(layer,kernel_array,stride_array,filter_array):
    x1 = Conv2D(kernel_size=kernel_array[0],strides=stride_array[0],filters=filter_array[0],padding='same')(layer)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    
    x2 = Conv2D(kernel_size=kernel_array[1][0],strides=stride_array[1][0],filters=filter_array[1][0],padding='same')(layer)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x21 = Conv2D(kernel_size=kernel_array[1][1],strides=stride_array[1][1],filters=filter_array[1][1],padding='same')(x2)
    x21 = BatchNormalization()(x21)
    x21 = ReLU()(x21)
    x22 = Conv2D(kernel_size=kernel_array[1][2],strides=stride_array[1][2],filters=filter_array[1][2],padding='same')(x2)
    x22 = BatchNormalization()(x22)
    x22 = ReLU()(x22)
    x2 = Concatenate(axis=-1)([x21,x22])
    
    x3 = Conv2D(kernel_size=kernel_array[2][0],strides=stride_array[2][0],filters=filter_array[2][0],padding='same')(layer)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Conv2D(kernel_size=kernel_array[2][1],strides=stride_array[2][1],filters=filter_array[2][1],padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x31 = Conv2D(kernel_size=kernel_array[2][2],strides=stride_array[2][2],filters=filter_array[2][2],padding='same')(x3)
    x31 = BatchNormalization()(x31)
    x31 = ReLU()(x31)
    x32 = Conv2D(kernel_size=kernel_array[2][3],strides=stride_array[2][3],filters=filter_array[2][3],padding='same')(x3)
    x32 = BatchNormalization()(x32)
    x32 = ReLU()(x32)
    x3 = Concatenate(axis=-1)([x31,x32])
    
    x4 = MaxPooling2D(pool_size=3,strides=1,padding='same')(layer)
    x4 = Conv2D(kernel_size=kernel_array[3],strides=stride_array[3],filters=filter_array[3],padding='same')(x4)
    x4 = BatchNormalization()(x4)
    x4 = ReLU()(x4)
    x = Concatenate(axis=-1)([x1,x2,x3,x4])
    return x