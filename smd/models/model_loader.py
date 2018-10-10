from b_lstm import create_b_lstm
import config
from keras import optimizers


def load_model(cfg):

    if cfg["model"] == "blstm":
        model = create_b_lstm()

    if cfg["optimizer"] == "SGD":
        optimizer = optimizers.SGD(lr=cfg["lr"], momentum=cfg["momentum"], decay=cfg["decay"])

    model.compile(optimizer, loss=config.LOSS, metrics=config.METRICS)
    return model
