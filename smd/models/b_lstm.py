from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed
from smd import config


def create_b_lstm(hidden_units, dropout=0.05):
    model = Sequential()

    i = 0
    for unit in hidden_units:
        if i == 0:
            model.add(Bidirectional(LSTM(unit, dropout=dropout, return_sequences=True)), input_shape=(None, config.N_MELS))
        else:
            model.add(Bidirectional(LSTM(unit, dropout=dropout, return_sequences=True)))
        i += 1

    model.add(TimeDistributed(Dense(config.CLASSES, activation='sigmoid')))

    return model
