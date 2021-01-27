# lstm model 
# Single shot predictions

from tensorflow.keras import layers, models

def lstm(units,drop,lookback,inp_features,future,out_features):

    input=layers.Input(shape=(lookback,10))
    result=layers.LSTM(units,dropout=drop,return_sequences=False)(input)
    result=layers.Dense(future*channel)(result)
    result=layers.Reshape(target_shape=(future,channel))(result)
    
    return models.Model(inputs=input,outputs=result)
