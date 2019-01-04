from smd import config
from smd.models.lstm import create_lstm
from smd.models.tcn import create_tcn
from smd.models.cldnn import create_cldnn
from keras import optimizers


def load_model(cfg):
    if cfg["type"] == "lstm":
        model = create_lstm(hidden_units=cfg["hidden_units"],
                            dropout=cfg["dropout"],
                            bidirectional=cfg["bidirectional"])
    elif cfg["type"] == "cldnn":
        model = create_cldnn(filters_list=cfg["filters_list"],
                             lstm_units=cfg["lstm_units"],
                             fc_units=cfg["fc_units"],
                             kernel_sizes=cfg["kernel_sizes"],
                             dropout=cfg["dropout"])
    elif cfg["type"] == "tcn":
        model = create_tcn(list_n_filters=cfg["list_n_filters"],
                           kernel_size=cfg["kernel_size"],
                           dilations=cfg["dilations"],
                           nb_stacks=cfg["nb_stacks"],
                           activation=cfg["activation"],
                           n_layers=cfg["n_layers"],
                           dropout_rate=cfg["dropout_rate"],
                           use_skip_connections=cfg["use_skip_connections"],
                           bidirectional=cfg["bidirectional"])
    else:
        raise ValueError(
            "Configuration error: the specified model is not yet implemented.")

    if cfg["optimizer"]["name"] == "SGD":
        optimizer = optimizers.SGD(
            lr=cfg["optimizer"]["lr"], momentum=cfg["optimizer"]["momentum"], decay=cfg["optimizer"]["decay"])
    elif cfg["optimizer"]["name"] == "adam":
        optimizer = optimizers.adam(lr=cfg["optimizer"]["lr"],
                                    beta_1=cfg["optimizer"]["beta_1"],
                                    beta_2=cfg["optimizer"]["beta_2"],
                                    epsilon=cfg["optimizer"]["epsilon"],
                                    decay=cfg["optimizer"]["decay"],
                                    clipnorm=cfg["optimizer"]["clipnorm"])
    else:
        raise ValueError(
            "Configuration error: the specified optimizer is not yet implemented.")

    model.compile(optimizer, loss=config.LOSS, metrics=config.METRICS)

    model.summary()
    return model


def compile_model(model, cfg):
    if cfg["optimizer"]["name"] == "SGD":
        optimizer = optimizers.SGD(
            lr=cfg["optimizer"]["lr"], momentum=cfg["optimizer"]["momentum"], decay=cfg["optimizer"]["decay"])
    elif cfg["optimizer"]["name"] == "adam":
        optimizer = optimizers.adam(lr=cfg["optimizer"]["lr"],
                                    beta_1=cfg["optimizer"]["beta_1"],
                                    beta_2=cfg["optimizer"]["beta_2"],
                                    epsilon=cfg["optimizer"]["epsilon"],
                                    decay=cfg["optimizer"]["decay"],
                                    clipnorm=cfg["optimizer"]["clipnorm"])
    else:
        raise ValueError(
            "Configuration error: the specified optimizer is not yet implemented.")

    model.compile(optimizer, loss=config.LOSS, metrics=config.METRICS)

    model.summary()
    return model
