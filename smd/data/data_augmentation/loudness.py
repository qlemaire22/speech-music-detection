import random
from smd import config


def random_loudness_audio(audio, factor=None):
    if factor is None:
        strength = config.MAX_LOUDNESS_DB * 2 * (random.random() - .5)
        factor = 10**(strength / 20.)
    return audio * factor


def random_loudness_spec(spec, factor=None):
    """ Based on https://github.com/f0k/ismir2018/blob/master/experiments/augment.py """
    if factor is None:
        strength = config.MAX_LOUDNESS_DB * 2 * (random.random() - .5)
        factor = 10**(strength / 20.)
    return spec * factor
