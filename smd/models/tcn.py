from tcn import TCN
from tcn.tcn import process_dilations
from keras.models import Input, Model
from keras.layers import Dense, Activation
from smd import config


def create_tcn(list_n_filters=[8],
               kernel_size=4,
               dilations=[1, 2],
               nb_stacks=1,
               activation='norm_relu',
               n_layers=1,
               dropout_rate=0.05,
               use_skip_connections=True,
               bidirectional=True):
    if bidirectional:
        padding = 'same'
    else:
        padding = 'causal'

    dilations = process_dilations(dilations)

    input_layer = Input(shape=(None, config.N_MELS))

    for i in range(n_layers):
        if i == 0:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation,
                    padding, use_skip_connections, dropout_rate, return_sequences=True)(input_layer)
        else:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation,
                    padding, use_skip_connections, dropout_rate, return_sequences=True, name="tcn" + str(i))(x)

    x = Dense(config.CLASSES)(x)
    x = Activation('sigmoid')(x)
    output_layer = x

    return Model(input_layer, output_layer)
