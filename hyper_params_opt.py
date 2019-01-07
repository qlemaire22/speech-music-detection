import numpy as np
import os
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from smd.models import tcn, lstm, cldnn
import smd.config as config
import smd.utils as utils
from smd.data.data_generator import DataGenerator
from smd.data.dataset_loader import DatasetLoader
from smd.data.data_augmentation import random_loudness_spec, random_filter_spec, block_mixing_spec, pitch_time_deformation_spec
from smd.data import preprocessing

from keras import optimizers
import time
import datetime
import csv
from keras import backend as K


def data():
    n_eval = 0
    cfg = {"dataset": ["ofai", "muspeak", "esc-50"],
           "data_location": "/Users/quentin/Computer/DataSet/Music/speech_music_detection/",
           "target_seq_length": 270,
           "batch_size": 16
           }

    def training_data_processing(spec_file, annotation_file, mean, std, spec_file2=None, annotation_file2=None):
        spec = np.load(spec_file)
        spec, stretching_rate = pitch_time_deformation_spec(spec)
        spec = random_filter_spec(spec)
        spec = random_loudness_spec(spec)
        label = preprocessing.get_label(
            annotation_file, spec.shape[1], stretching_rate=stretching_rate)

        if not(spec_file2 is None):
            spec2 = np.load(spec_file2)
            spec2, stretching_rate2 = pitch_time_deformation_spec(spec2)
            spec2 = random_filter_spec(spec2)
            spec2 = random_loudness_spec(spec2)
            label2 = preprocessing.get_label(
                annotation_file2, spec2.shape[1], stretching_rate=stretching_rate2)
            spec, label = block_mixing_spec(spec, spec2, label, label2)

        mels = preprocessing.get_scaled_mel_bands(spec)
        mels = preprocessing.normalize(mels, mean, std)
        return mels, label

    def validation_data_processing(spec_file, annotation_file, mean, std):
        spec = np.load(spec_file)

        mels = preprocessing.get_scaled_mel_bands(spec)
        mels = preprocessing.normalize(mels, mean, std)
        n_frame = mels.shape[1]
        label = preprocessing.get_label(
            annotation_file, n_frame, stretching_rate=1)
        return mels, label

    datasets_config = utils.load_json("datasets.json")
    dataset = DatasetLoader(
        cfg["dataset"], cfg["data_location"], datasets_config)

    train_set = DataGenerator(dataset.get_train_set(),
                              cfg["batch_size"],
                              cfg["target_seq_length"],
                              training_data_processing,
                              dataset.get_training_mean(),
                              dataset.get_training_std(),
                              set_type="train")

    val_set = DataGenerator(dataset.get_val_set(),
                            cfg["batch_size"],
                            cfg["target_seq_length"],
                            validation_data_processing,
                            dataset.get_training_mean(),
                            dataset.get_training_std(),
                            set_type="val")
    return train_set, val_set


def fit_lstm(train_set, val_set):
    global n_eval

    n_eval += 1
    print(str(datetime.datetime.now()) + ": iteration number: " + str(n_eval))

    with open("hyper_opt_result.txt", 'a') as f:
        f.write(str(datetime.datetime.now()) +
                ": iteration number: " + str(n_eval) + "\n")

    cfg = {"optimizer":
           {
               "name": "SGD",
               "lr": 0.001,
               "momentum": 0.9,
               "decay": 0
           },
           "workers": 8,
           "use_multiprocessing": True,
           "n_epochs": 5
           }
    n_layer = {{choice([1, 2, 3, 4])}}
    layers = []

    layers.append(
        {{choice([25, 50, 75, 100, 125, 150, 175, 200, 225, 250])}})
    if conditional(n_layer) >= 2:
        layers.append(
            {{choice([25, 50, 75, 100, 125, 150, 175, 200, 225, 250])}})
        if conditional(n_layer) >= 3:
            layers.append(
                {{choice([25, 50, 75, 100, 125, 150, 175, 200, 225, 250])}})
            if conditional(n_layer) >= 4:
                layers.append(
                    {{choice([25, 50, 75, 100, 125, 150, 175, 200, 225, 250])}})

    model = lstm.create_lstm(hidden_units=layers, dropout={
                             {uniform(0.05, 0.5)}}, bidirectional=True)

    n_params = model.count_params()
    print(n_params)
    print(space)

    with open("hyper_opt_result.txt", 'a') as f:
        f.write(str(n_params))
        for key in space.keys():
            f.write(str(key) + " " + str(space[key]) + " ")
            f.write('\n')

    optimizer = optimizers.SGD(
        lr=cfg["optimizer"]["lr"], momentum=cfg["optimizer"]["momentum"], decay=cfg["optimizer"]["decay"])

    model.compile(loss=config.LOSS, metrics=config.METRICS,
                  optimizer=optimizer)

    result = model.fit_generator(train_set,
                                 epochs=cfg["n_epochs"],
                                 validation_data=val_set,
                                 workers=cfg["workers"],
                                 use_multiprocessing=cfg["use_multiprocessing"],
                                 shuffle=True,
                                 verbose=0
                                 )

    validation_loss = np.amin(result.history['val_loss'])

    csv_line = [n_eval, n_params, validation_loss]
    for key in space.keys():
        csv_line.append(space[key])
    with open(r'fit_b_lstm_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(csv_line)

    print('Best validation acc of epoch:', validation_loss)
    del model
    K.clear_session()
    return {'loss': validation_loss, 'status': STATUS_OK}


def fit_cldnn(train_set, val_set):
    global n_eval

    n_eval += 1
    print(str(datetime.datetime.now()) + ": iteration number: " + str(n_eval))

    with open("hyper_opt_result.txt", 'a') as f:
        f.write(str(datetime.datetime.now()) +
                ": iteration number: " + str(n_eval) + "\n")

    cfg = {"optimizer":
           {
               "name": "SGD",
               "lr": 0.001,
               "momentum": 0.9,
               "decay": 0
           },
           "workers": 8,
           "use_multiprocessing": True,
           "n_epochs": 5
           }

    n_layer_c = {{choice([1, 2, 3])}}
    n_layer_l = {{choice([1, 2, 3])}}
    n_layer_d = {{choice([1, 2, 3])}}

    filters_list = []
    lstm_units = []
    fc_units = []
    kernel_sizes = []

    filters_list.append({{choice([8, 16, 32, 64, 128])}})
    kernel_sizes.append({{choice([3, 5, 9, 11])}})
    if conditional(n_layer_c) >= 2:
        filters_list.append({{choice([8, 16, 32, 64, 128])}})
        kernel_sizes.append({{choice([3, 5, 9, 11])}})
        if conditional(n_layer_c) >= 3:
            filters_list.append({{choice([8, 16, 32, 64, 128])}})
            kernel_sizes.append({{choice([3, 5, 9, 11])}})

    lstm_units.append({{choice([25, 50, 75, 100, 125, 150])}})
    if conditional(n_layer_l) >= 2:
        lstm_units.append({{choice([25, 50, 75, 100, 125, 150])}})
        if conditional(n_layer_l) >= 3:
            lstm_units.append({{choice([25, 50, 75, 100, 125, 150])}})

    fc_units.append({{choice([25, 50, 75, 100, 125, 150])}})
    if conditional(n_layer_d) >= 2:
        fc_units.append({{choice([25, 50, 75, 100, 125, 150])}})
        if conditional(n_layer_d) >= 3:
            fc_units.append({{choice([25, 50, 75, 100, 125, 150])}})

    model = cldnn.create_cldnn(filters_list=filters_list,
                               lstm_units=lstm_units,
                               fc_units=fc_units,
                               kernel_sizes=kernel_sizes,
                               dropout={{uniform(0.05, 0.5)}},
                               bidirectional=False)
    n_params = model.count_params()
    print(n_params)
    print(space)

    with open("hyper_opt_result.txt", 'a') as f:
        f.write(str(n_params))
        for key in space.keys():
            f.write(str(key) + " " + str(space[key]) + " ")
            f.write('\n')

    optimizer = optimizers.SGD(
        lr=cfg["optimizer"]["lr"], momentum=cfg["optimizer"]["momentum"], decay=cfg["optimizer"]["decay"])

    model.compile(loss=config.LOSS, metrics=config.METRICS,
                  optimizer=optimizer)

    result = model.fit_generator(train_set,
                                 epochs=cfg["n_epochs"],
                                 validation_data=val_set,
                                 workers=cfg["workers"],
                                 use_multiprocessing=cfg["use_multiprocessing"],
                                 shuffle=True,
                                 verbose=0
                                 )

    validation_loss = np.amin(result.history['val_loss'])

    csv_line = [n_eval, n_params, validation_loss]
    for key in space.keys():
        csv_line.append(space[key])
    with open(r'fit_cldnn_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(csv_line)

    print('Best validation acc of epoch:', validation_loss)
    del model
    K.clear_session()
    return {'loss': validation_loss, 'status': STATUS_OK}


def fit_tcn(train_set, val_set):
    global n_eval

    n_eval += 1
    print(str(datetime.datetime.now()) + ": iteration number: " + str(n_eval))

    with open("hyper_opt_result.txt", 'a') as f:
        f.write(str(datetime.datetime.now()) +
                ": iteration number: " + str(n_eval) + "\n")

    cfg = {"optimizer":
           {
               "name": "SGD",
               "lr": 0.001,
               "momentum": 0.9,
               "decay": 0
           },
           "workers": 8,
           "use_multiprocessing": True,
           "n_epochs": 5
           }
    nb_filters = []
    kernel_size = {{choice([3, 5, 7, 9, 11, 13, 15, 17, 19])}}
    dilations = {{choice([[2 ** i for i in range(4)],
                          [2 ** i for i in range(5)],
                          [2 ** i for i in range(6)],
                          [2 ** i for i in range(7)],
                          [2 ** i for i in range(8)],
                          [2 ** i for i in range(9)]])}}
    nb_stacks = {{choice([3, 4, 5, 6, 7, 8, 9, 10])}}
    use_skip_connections = {{choice([True, False])}}
    n_layers = {{choice([1, 2, 3, 4])}}

    nb_filters.append({{choice([8, 16, 32])}})
    if conditional(n_layers) >= 2:
        nb_filters.append({{choice([8, 16, 32])}})
        if conditional(n_layers) >= 3:
            nb_filters.append({{choice([8, 16, 32])}})
            if conditional(n_layers) >= 4:
                nb_filters.append({{choice([8, 16, 32])}})

    model = tcn.create_tcn(list_n_filters=nb_filters,
                           kernel_size=kernel_size,
                           dilations=dilations,
                           nb_stacks=nb_stacks,
                           n_layers=n_layers,
                           use_skip_connections=use_skip_connections,
                           dropout_rate={{uniform(0.05, 0.5)}},
                           bidirectional=False)

    n_params = model.count_params()
    print(n_params)
    print(space)

    with open("hyper_opt_result.txt", 'a') as f:
        f.write(str(n_params))
        for key in space.keys():
            f.write(str(key) + " " + str(space[key]) + " ")
            f.write('\n')

    optimizer = optimizers.SGD(
        lr=cfg["optimizer"]["lr"], momentum=cfg["optimizer"]["momentum"], decay=cfg["optimizer"]["decay"])

    model.compile(loss=config.LOSS, metrics=config.METRICS,
                  optimizer=optimizer)

    result = model.fit_generator(train_set,
                                 epochs=cfg["n_epochs"],
                                 validation_data=val_set,
                                 workers=cfg["workers"],
                                 use_multiprocessing=cfg["use_multiprocessing"],
                                 shuffle=True,
                                 verbose=0
                                 )

    validation_loss = np.amin(result.history['val_loss'])

    csv_line = [n_eval, n_params, validation_loss]
    for key in space.keys():
        csv_line.append(space[key])
    with open(r'fit_tcn_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(csv_line)

    print('Best validation acc of epoch:', validation_loss)
    del model
    K.clear_session()
    return {'loss': validation_loss, 'status': STATUS_OK}


def fit_nc_tcn(train_set, val_set):
    global n_eval

    n_eval += 1
    print(str(datetime.datetime.now()) + ": iteration number: " + str(n_eval))

    with open("hyper_opt_result.txt", 'a') as f:
        f.write(str(datetime.datetime.now()) +
                ": iteration number: " + str(n_eval) + "\n")

    cfg = {"optimizer":
           {
               "name": "adam",
               "lr": 0.002,
               "clipnorm": 1,
               "beta_1": 0.9,
               "beta_2": 0.999,
               "epsilon": None,
               "decay": 0.0
           },
           "workers": 8,
           "use_multiprocessing": True,
           "n_epochs": 7
           }
    nb_filters = []
    kernel_size = {
        {choice([11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41])}}
    dilations = {{choice([[2 ** i for i in range(4)],
                          [2 ** i for i in range(5)],
                          [2 ** i for i in range(6)],
                          [2 ** i for i in range(7)],
                          [2 ** i for i in range(8)],
                          [2 ** i for i in range(9)],
                          [2 ** i for i in range(10)]])}}
    nb_stacks = {{choice([3, 4, 5, 6, 7, 8, 9, 10])}}
    use_skip_connections = False
    n_layers = {{choice([1, 2, 3, 4, 5])}}

    nb_filters.append({{choice([8, 16, 32, 64])}})
    if conditional(n_layers) >= 2:
        nb_filters.append({{choice([8, 16, 32, 64])}})
        if conditional(n_layers) >= 3:
            nb_filters.append({{choice([8, 16, 32, 64])}})
            if conditional(n_layers) >= 4:
                nb_filters.append({{choice([8, 16, 32, 64])}})
                if conditional(n_layers) >= 5:
                    nb_filters.append({{choice([8, 16, 32, 64])}})

    model = tcn.create_tcn(list_n_filters=nb_filters,
                           kernel_size=kernel_size,
                           dilations=dilations,
                           nb_stacks=nb_stacks,
                           n_layers=n_layers,
                           use_skip_connections=use_skip_connections,
                           dropout_rate={{uniform(0.01, 0.25)}},
                           bidirectional=True)

    n_params = model.count_params()
    print(n_params)
    print(space)

    with open("hyper_opt_result.txt", 'a') as f:
        f.write(str(n_params))
        for key in space.keys():
            f.write(str(key) + " " + str(space[key]) + " ")
            f.write('\n')

    optimizer = optimizers.adam(
        lr=cfg["optimizer"]["lr"], clipnorm=cfg["optimizer"]["clipnorm"])

    model.compile(loss=config.LOSS, metrics=config.METRICS,
                  optimizer=optimizer)

    result = model.fit_generator(train_set,
                                 epochs=cfg["n_epochs"],
                                 validation_data=val_set,
                                 workers=cfg["workers"],
                                 use_multiprocessing=cfg["use_multiprocessing"],
                                 shuffle=True,
                                 verbose=0
                                 )

    validation_loss = np.amin(result.history['val_loss'])

    csv_line = [n_eval, n_params, validation_loss]
    for key in space.keys():
        csv_line.append(space[key])
    with open(r'fit_b_tcn_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(csv_line)

    print('Best validation acc of epoch:', validation_loss)
    del model
    K.clear_session()
    return {'loss': validation_loss, 'status': STATUS_OK}


if __name__ == '__main__':
    MAX_EVALS = 500

    if not(os.path.isdir("checkpoint")):
        os.makedirs("checkpoint")
        print("Checkpoint folder created.")

    t0 = time.time()

    best_run = optim.minimize(model=fit_cldnn,
                              data=data,
                              algo=tpe.suggest,
                              max_evals=MAX_EVALS,
                              rseed=None,
                              trials=Trials())

    t1 = time.time()
    print("Number of evaluations: " + str(MAX_EVALS))
    print("Total time: " + str(t1 - t0))
    print("Time by eval: " + str((t1 - t0) / MAX_EVALS))
    print("Best found values:")
    print(best_run)

    with open("hyper_opt_result.txt", 'a') as f:
        f.write("Number of evaluations: " + str(MAX_EVALS) + "\n")
        f.write("Total time: " + str(t1 - t0) + "\n")
        f.write("Time by eval: " + str((t1 - t0) / MAX_EVALS) + "\n")
        f.write("Best found values:" + "\n")
        for key in best_run[0].keys():
            f.write(str(key) + " " + str(best_run[key]) + "\n")
