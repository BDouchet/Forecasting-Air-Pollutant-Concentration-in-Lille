# Seq2seq rnn with lstm cells

from tensorflow.keras import layers, models

def s2sL(units,bn,drop=0.0,channel=3):
    encoder_inputs = layers.Input(shape=(168,10))

    encoder = layers.LSTM(units, dropout=drop,return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    if bn:
       encoder_outputs=BatchNormalization()(encoder_outputs)
       state_c=BatchNormalization()(state_c)
       
    decoder=layers.RepeatVector(48)(encoder_outputs)
    decoder_lstm = layers.LSTM(units, dropout=drop, return_sequences=True, return_state=False)
    decoder = decoder_lstm(decoder, initial_state=[encoder_outputs, state_c])
    
    out = layers.TimeDistributed(Dense(channel))(decoder)
    return models.Model(encoder_inputs, out)
