import librosa
import random
from smd.data.data_augmentation import config
import smd.data.preprocessing.config


def pitch_shifting(audio, n_steps=None):
    if n_steps is None:
        n_steps = float(random.randint(int(config.SHIFTING_MIN * 2), int(config.SHIFTING_MAX * 2))) / 2
    return librosa.effects.pitch_shift(audio, smd.data.preprocessing.config.SAMPLING_RATE, n_steps=n_steps)
