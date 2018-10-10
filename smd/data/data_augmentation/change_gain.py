import random
from smd.data.data_augmentation import config


def change_gain(audio):
    rate = 3
    print(rate)
    return audio - rate
