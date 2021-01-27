# Prepocess dataset and train a model

import numpy as np
from tensorflow.keras import models,layers
import tensorflow as tf


# Time Series Preprocessing

from preprocess import truncate

x_train,y_train,x_val,y_val,x_test,y_test=truncate(data,lookback=168,forecast=48,len_test=10000,len_val=10000,target='full')
print(x_train.shape,y_train.shape,x_val.shape,y_val.shape,x_test.shape,y_test.shape)

# -> (47800, 168, 10) (47800, 48, 3) (10000, 168, 10) (10000, 48, 3) (10000, 168, 10) (10000, 48, 3)


# Model Generation (Basic LSTM)

model=models.Sequential()
model.add(layers.Input(shape=(168,10))
model.add(layers.LSTM(16,dropout=drop,return_sequences=False))
model.add(layers.Dense(48*3))
model.add(layers.Reshape(target_shape=(48,3)))


#Training

from tensorflow.keras.callbacks import Tensorboard
name='model_demo'
path='a_path'

tensorboard = TensorBoard(logdir+"logs/{}".format(name),histogram_freq=1)
model.compile(loss='mse',optimizer=tf.keras.optimizers.RMSprop(0.001),metrics=['mae'])
model.fit(x_train,
          y_train,
          epochs=30,
          batch_size=128,
          validation_data=(x_val,y_val),
          callbacks=[tensorboard])
model.save(path+name+'.h5',overwrite=True)
