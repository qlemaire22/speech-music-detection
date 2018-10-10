from smd.models import config
from smd.models.b_lstm import create_b_lstm
from keras import optimizers


def load_model(cfg):

    if cfg["model"] == "blstm":
        model = create_b_lstm()

    if cfg["optimizer"] == "SGD":
        optimizer = optimizers.SGD(lr=cfg["lr"], momentum=cfg["momentum"], decay=cfg["decay"])

    model.compile(optimizer, loss=config.LOSS, metrics=config.METRICS)
    return model
