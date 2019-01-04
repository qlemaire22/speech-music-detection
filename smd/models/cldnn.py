from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Conv2D, Activation, Bidirectional, Reshape, Dropout, MaxPooling2D
from smd import config


def create_cldnn(filters_list=[32], lstm_units=[50], fc_units=[50], kernel_sizes=[15], dropout=0.05, bidirectional=False):
    model = Sequential()

    model.add(Reshape((-1, config.N_MELS, 1), input_shape=(None, config.N_MELS)))

    for filters, kernel_size in zip(filters_list, kernel_sizes):
        model.add(Conv2D(filters, kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(MaxPooling2D(pool_size=(1, 2)))

    _, _, sx, sy = model.layers[-1].output.shape

    model.add(Reshape((-1, int(sx * sy))))

    if bidirectional:
        for unit in lstm_units:
            model.add(Bidirectional(LSTM(unit, dropout=dropout, return_sequences=True)))
    else:
        for unit in lstm_units:
            model.add(LSTM(unit, dropout=dropout, return_sequences=True))

    for units in fc_units:
        model.add(Dense(units, activation='relu'))

    model.add(TimeDistributed(Dense(config.CLASSES, activation='sigmoid')))

    return model
