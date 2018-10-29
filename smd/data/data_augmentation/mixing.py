import random
from smd import config
import numpy as np


def block_mixing_audio(audio1, audio2, overlap=None):
    n1 = len(audio1)
    n2 = len(audio2)

    b1 = int(config.BLOCK_MIXING_MIN * min(n1, n2))
    b2 = int(config.BLOCK_MIXING_MAX * min(n1, n2))

    if overlap is None:
        overlap = random.randint(b1, b2)
    else:
        overlap = int(overlap * min(n1, n2))

    new_audio = np.zeros(n1 + n2 - overlap)

    new_audio[:n1] += audio1
    new_audio[n1 - overlap:] += audio2

    return new_audio


def block_mixing_spec(spec1, spec2, label1, label2, overlap=None):
    feat, n1 = spec1.shape
    n2 = spec2.shape[1]

    b1 = int(config.BLOCK_MIXING_MIN * min(n1, n2))
    b2 = int(config.BLOCK_MIXING_MAX * min(n1, n2))

    if overlap is None:
        overlap = random.randint(b1, b2)
    else:
        overlap = int(overlap * min(n1, n2))

    new_spec = np.zeros((feat, n1 + n2 - overlap))
    new_label = np.zeros((config.CLASSES, n1 + n2 - overlap))

    new_spec[:, :n1] += spec1
    new_spec[:, -n2:] += spec2

    new_label[:, :n1] += label1
    new_label[:, -n2:] += label2
    new_label = np.minimum(new_label, 1)

    return new_spec, new_label
