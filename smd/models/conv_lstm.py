from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Conv1D, Activation, Bidirectional
from smd import config


def create_conv_lstm(hidden_units=[50], filters=32, kernel_size=15, dropout=0.05, bidirectional=True):
    model = Sequential()

    model.add(Conv1D(filters, kernel_size), input_shape=(None, config.N_MELS))
    model.add(Activation('relu'))

    if bidirectional:
        for unit in hidden_units:
            model.add(Bidirectional(LSTM(unit, dropout=dropout, return_sequences=True)))
    else:
        for unit in hidden_units:
            model.add(LSTM(unit, dropout=dropout, return_sequences=True))

    model.add(TimeDistributed(Dense(config.CLASSES, activation='sigmoid')))

    return model
