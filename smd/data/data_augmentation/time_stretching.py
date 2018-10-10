import librosa
import random
from smd.data.data_augmentation import config


def time_stretching(audio, rate=None):
    if rate is None:
        rate = random.uniform(config.STRETCHING_MIN, config.STRETCHING_MAX)
    return librosa.effects.time_stretch(audio, rate), rate
