# Seq2seq rnn with lstm cells

from tensorflow.keras import layers, models

def s2sL(units,bn,drop=0.0,channel=3):
    encoder_inputs = Input(shape=(168,10))

    encoder = GRU(units, dropout=drop,return_state=True)
    _,encoder_states = encoder(encoder_inputs)
    if bn:
       encoder_states=BatchNormalization()(encoder_states)
    decoder=RepeatVector(48)(encoder_states)
    decoder_gru = GRU(units, dropout=drop, return_sequences=True, return_state=False)
    decoder = decoder_gru(decoder, initial_state=encoder_states)
    
    out = TimeDistributed(Dense(channel))(decoder)
    return models.Model(encoder_inputs, out)
