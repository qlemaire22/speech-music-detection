from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional
from smd import config


def create_lstm(hidden_units=[50], dropout=0.05, bidirectional=True):
    model = Sequential()

    if bidirectional:
        i = 0
        for unit in hidden_units:
            if i == 0:
                model.add(Bidirectional(LSTM(unit, dropout=dropout, return_sequences=True), input_shape=(None, config.N_MELS)))
            else:
                model.add(Bidirectional(LSTM(unit, dropout=dropout, return_sequences=True)))
            i += 1
    else:
        i = 0
        for unit in hidden_units:
            if i == 0:
                model.add(LSTM(unit, dropout=dropout, return_sequences=True), input_shape=(None, config.N_MELS))
            else:
                model.add(LSTM(unit, dropout=dropout, return_sequences=True))
            i += 1

    model.add(TimeDistributed(Dense(config.CLASSES, activation='sigmoid')))

    return model
