from smd.models import config
from smd.models.b_lstm import create_b_lstm
from smd.models.tcn import create_tcn
from keras import optimizers


def load_model(cfg):

    if cfg["model"] == "blstm":
        model = create_b_lstm(hidden_units=cfg["hidden_units"],
                              dropout=cfg["dropout"])
    elif cfg["model"] == "bconvlstm":
        None
    elif cfg["model"] == "tcn":
        model = create_tcn(nb_filters=cfg["nb_filters"],
                           kernel_size=cfg["kernel_size"],
                           dilations=cfg["dilations"],
                           nb_stacks=cfg["nb_stacks"],
                           activation=cfg["activation"],
                           use_skip_connections=cfg["use_skip_connections"],
                           dropout_rate=cfg["dropout_rate"])
    else:
        raise ValueError(
            "Configuration error: the specified model is not implemented.")

    if cfg["optimizer"]["name"] == "SGD":
        optimizer = optimizers.SGD(
            lr=cfg["optimizer"]["lr"], momentum=cfg["optimizer"]["momentum"], decay=cfg["optimizer"]["decay"])

    model.compile(optimizer, loss=config.LOSS, metrics=config.METRICS)
    return model
