from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D, Bidirectional, TimeDistributed, Reshape
import smd.data.preprocessing.config
from smd.models import config


def create_b_conv_lstm(filters_list, kernel_size_list, stride_list, dilation_rate_list, dropout=0.05):
    model = Sequential()

    model.add(Reshape((-1, 1, 1, smd.data.preprocessing.config.N_MELS), input_shape=(None, smd.data.preprocessing.config.N_MELS)))

    for filters, kernel_size, strides, dilation_rate in filters_list, kernel_size_list, stride_list, dilation_rate_list:
        model.add(Bidirectional(ConvLSTM2D(filters,
                                           kernel_size,
                                           strides=strides,
                                           padding='same',
                                           data_format='channels_last',
                                           dilation_rate=dilation_rate,
                                           return_sequences=True,
                                           dropout=dropout)))

    model.add(TimeDistributed(Dense(config.CLASSES, activation='sigmoid')))

    return model
