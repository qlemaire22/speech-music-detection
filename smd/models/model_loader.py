from smd import config
from smd.models.b_lstm import create_b_lstm
from smd.models.tcn import create_tcn
from smd.models.b_tcn import create_b_tcn
from smd.models.b_conv_lstm import create_b_conv_lstm
from keras import optimizers


def load_model(cfg):

    if cfg["type"] == "blstm":
        model = create_b_lstm(hidden_units=cfg["hidden_units"],
                              dropout=cfg["dropout"])
    elif cfg["type"] == "bconvlstm":
        model = create_b_conv_lstm(cfg["filters_list"],
                                   cfg["kernel_size_list"],
                                   cfg["stride_list"],
                                   cfg["dilation_rate_list"],
                                   cfg["dropout"])
    elif cfg["type"] == "tcn":
        model = create_tcn(list_n_filters=cfg["list_n_filters"],
                           kernel_size=cfg["kernel_size"],
                           dilations=cfg["dilations"],
                           nb_stacks=cfg["nb_stacks"],
                           activation=cfg["activation"],
                           n_layers=cfg["n_layers"],
                           use_skip_connections=cfg["use_skip_connections"],
                           dropout_rate=cfg["dropout_rate"])
    elif cfg["type"] == "btcn":
        model = create_b_tcn(list_n_filters=cfg["list_n_filters"],
                             kernel_size=cfg["kernel_size"],
                             dilations=cfg["dilations"],
                             nb_stacks=cfg["nb_stacks"],
                             activation=cfg["activation"],
                             n_layers=cfg["n_layers"],
                             use_skip_connections=cfg["use_skip_connections"],
                             dropout_rate=cfg["dropout_rate"])
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
