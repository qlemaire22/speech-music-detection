from tcn import TCN
from tcn.tcn import process_dilations
from keras.models import Input, Model
from keras.layers import Dense, Activation
from smd import config


def create_tcn(nb_filters=32, kernel_size=4, dilations=[1, 2, 4], nb_stacks=1, activation='norm_relu', use_skip_connections=True, dropout_rate=0.05):
    dilations = process_dilations(dilations)

    input_layer = Input(shape=(None, config.N_MELS))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, activation,
            use_skip_connections, dropout_rate, return_sequences=True)(input_layer)

    x = Dense(config.CLASSES)(x)
    x = Activation('sigmoid')(x)
    output_layer = x

    return Model(input_layer, output_layer)
