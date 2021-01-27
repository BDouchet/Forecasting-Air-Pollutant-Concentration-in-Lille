# gru model 
# Single shot predictions

from tensorflow.keras import layers, models

def gru(units,drop,out_features):

    input=layers.Input(shape=(168,10))
    result=layers.GRU(units,dropout=drop,return_sequences=False)(input)
    result=layers.Dense(future*channel)(result)
    result=layers.Reshape(target_shape=(48,out-features))(result)
    
    return models.Model(inputs=input,outputs=result)
