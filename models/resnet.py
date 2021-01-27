# ResNet network 
# 24 weight layers

from tensorflow.keras.layers import Input, Conv1D, Dropout, BatchNormalization, AveragePooling1D, Reshape, Flatten
from tensorflow.keras import models

def resblock(x, kernelsize, depth,bn,dropout):
    y=layers.Conv1D(depth,kernelsize,strides=1,activation='linear',padding='same')(x)
    y=layers.BatchNormalization()(y)
    y=layers.Conv1D(depth,kernelsize,strides=1,padding='same')(y)
    y=layers.Add()([x,y])
    if bn:
        y=layers.BatchNormalization()(y)
    y=Dropout(dropout)(y)
    return y

def Resnet(units,kernel,dropout,bn=True,out=3,activation='linear'):
    input=Input(shape=(168,10))
    result=Conv1D(units,kernel,activation=activation,padding='same')(input)
    if bn:
        result=BatchNormalization()(result)
    result=Dropout(dropout)(result)
    result=AveragePooling1D()(result)
    result=resblock(result,kernel,units,bn,dropout)
    result=resblock(result,kernel,units,bn,dropout)
    result=resblock(result,kernel,units,bn,dropout)
    result=AveragePooling1D()(result)
    
    result=Conv1D(units*2,kernel,activation=activation,padding='same')(result)
    if bn:
        result=BatchNormalization()(result)
    result=Dropout(dropout)(result)
    result=resblock(result,kernel,units*2,bn,dropout)
    result=resblock(result,kernel,units*2,bn,dropout)
    result=resblock(result,kernel,units*2,bn,dropout)
    result=AveragePooling1D()(result)
    
    result=Conv1D(units*4,kernel,activation=activation,padding='same')(result)
    if bn:
        result=BatchNormalization()(result)
    result=Dropout(dropout)(result)
    result=resblock(result,kernel,units*4,bn,dropout)
    result=resblock(result,kernel,units*4,bn,dropout)
    result=resblock(result,kernel,units*4,bn,dropout)
    result=AveragePooling1D()(result)
    
    result=Conv1D(units*2,1,activation=activation,padding='same')(result)
    result=Flatten()(result)
    output=Dense(48*out,activation=activation)(result)
    if out !=1:
        output=Reshape(target_shape=(48,out))(output)
        
    return models.Model(inputs=input,outputs=output)
