import random
from smd.data.data_augmentation import config


def change_gain(audio, rate=None):
    if rate is None:
        rate = random.uniform(config.GAIN_MIN, config.GAIN_MAX)
    return audio * rate
