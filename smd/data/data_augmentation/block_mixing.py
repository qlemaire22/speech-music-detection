import random
from smd.data.data_augmentation import config
import numpy as np


def block_mixing(audio1, audio2):
    n1 = len(audio1)
    n2 = len(audio2)

    b1 = int(config.BLOCK_MIXING_MIN * min(n1, n2))
    b2 = int(config.BLOCK_MIXING_MAX * min(n1, n2))

    overlap = random.randint(b1, b2)

    new_audio = np.zeros(n1 + n2 - overlap)

    new_audio[:n1] += audio1
    new_audio[n1 - overlap:] += audio2

    return new_audio
