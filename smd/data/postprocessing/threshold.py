import numpy as np


def apply_threshold(output, speech_threshold=0.5, music_threshold=0.5):
    output[0] = np.where(output[0] >= speech_threshold, 1, 0)
    output[1] = np.where(output[1] >= music_threshold, 1, 0)
    return output.astype(int)
