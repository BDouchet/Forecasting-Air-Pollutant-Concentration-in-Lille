# VGG network 
# 9 convolutional layers

from tensorflow.keras.layers import Input, Conv1D, Dropout, BatchNormalization, AveragePooling1D, Reshape, Flatten
from tensorflow.keras import models


def VGG(units,kernel=5,dropout=0.0,bn=True,channel=3):
    model=models.Sequential()

    model.add(Conv1D(units, kernel,input_shape=(168,10),activation='linear',padding='same'))
    if bn:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Conv1D(units, kernel,activation='linear',padding='same'))
    if bn:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(AveragePooling1D())

    model.add(Conv1D(units*2, kernel,activation='linear',padding='same'))
    if bn:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Conv1D(units*2, kernel,activation='linear',padding='same'))
    if bn:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(AveragePooling1D())

    model.add(Conv1D(units*4, kernel,activation='linear',padding='same'))
    if bn:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Conv1D(units*4, kernel,activation='linear',padding='same'))
    if bn:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Conv1D(units,1,activation='linear',padding='same'))
    model.add(Flatten())
    model.add(Dense(units*4,activation='linear'))
    model.add(Dense(48*3,activation='linear'))
    model.add(Reshape(target_shape=(48,channel)))
    
    return model
