from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed
import smd.data.preprocessing.config
import config


def create_b_lstm(hidden_units=[100, 100, 100], dropout=0.2):
    model = Sequential()

    i = 0
    for unit in hidden_units:
        if i == 0:
            model.add(Bidirectional(LSTM(unit, dropout=0.2, return_sequences=True)), input_shape=(None, smd.data.preprocessing.config.N_MELS))
        else:
            model.add(Bidirectional(LSTM(unit, dropout=0.2, return_sequences=True)))
        i += 1

    model.add(TimeDistributed(Dense(config.CLASSES, activation='sigmoid')))

    return model
