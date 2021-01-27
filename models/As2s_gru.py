# Seq2seq rnn with Luong Attention and lstm cells

from tensorflow.keras import layers, models

def As2sG(units,bn,dropout=0.0,channel=3):
    encoder_inputs = layers.Input(shape=(168,10))

    encoder = layers.GRU(units, dropout=dropout,return_state=True,return_sequences=True)
    stack_h, state_h = encoder(encoder_inputs)
    if bn:
        state_h=layers.BatchNormalization()(state_h)
       
    decoder=layers.RepeatVector(48)(state_h)
    decoder_lstm = layers.GRU(units, return_sequences=True, return_state=False)
    decoder_h = decoder_lstm(decoder, initial_state=state_h)
    
    attention = layers.dot([decoder_h, stack_h],axes=[2,2])
    attention = layers.Activation('softmax')(attention)

    context = layers.dot([attention, stack_h], axes=[2,1])
    if bn :
        context=layers.BatchNormalization()(context)

    decoder_combined_context = layers.concatenate([context, decoder_h])

    out = TimeDistributed(layers.Dense(channel))(decoder_combined_context)
    return models.Model(encoder_inputs, out)
