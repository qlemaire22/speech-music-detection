import librosa
import random
from smd import config
from scipy.ndimage import affine_transform, spline_filter
import numpy as np


def pitch_shifting_audio(audio, n_steps=None):
    if n_steps is None:
        n_steps = float(random.randint(int(config.SHIFTING_MIN * 2), int(config.SHIFTING_MAX * 2))) / 2
    return librosa.effects.pitch_shift(audio, config.SAMPLING_RATE, n_steps=n_steps)


def time_stretching_audio(audio, rate=None):
    if rate is None:
        rate = random.uniform(config.STRETCHING_MIN, config.STRETCHING_MAX)
    return librosa.effects.time_stretch(audio, rate), rate


def pitch_shifting_spec(spec, n_steps=None):
    if n_steps is None:
        n_steps = float(random.randint(int(config.SHIFTING_MIN * 2), int(config.SHIFTING_MAX * 2))) / 2

    spec = spline_filter(spec, 2).astype(np.float32)
    spec = affine_transform(spec, (1 / stretch, 1 / shift),
                            output_shape=(keep_frames, keep_bins),
                            offset=offset, mode='constant', order=2,
                            prefilter=False)
    return np.maximum(spec, 0)
