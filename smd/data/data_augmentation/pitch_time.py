import librosa
import random
from smd import config
from scipy.ndimage import affine_transform, spline_filter
import numpy as np
import warnings


def pitch_shifting_audio(audio, n_steps=None):
    if n_steps is None:
        n_steps = float(random.randint(int(config.SHIFTING_MIN * 2), int(config.SHIFTING_MAX * 2))) / 2
    return librosa.effects.pitch_shift(audio, config.SAMPLING_RATE, n_steps=n_steps)


def time_stretching_audio(audio, rate=None):
    if rate is None:
        rate = random.uniform(config.STRETCHING_MIN, config.STRETCHING_MAX)
    return librosa.effects.time_stretch(audio, rate), rate


def pitch_time_deformation_spec(spec, stretch_rate=None, shift_rate=None):
    """ Based on https://github.com/f0k/ismir2018/blob/master/experiments/augment.py """
    shape = spec.shape
    random = (np.random.rand(2) - .5) * 2

    warnings.filterwarnings("ignore", "The behaviour of affine_transform with a "
                            "one-dimensional array supplied for the matrix parameter "
                            "has changed in scipy 0.18.0.", module="scipy.ndimage")

    if stretch_rate is None:
        stretch_rate = 1 + random[0] * config.MAX_STRETCHING

    if shift_rate is None:
        shift_rate = 1 + random[1] * config.MAX_SHIFTING_PITCH

    new_length = int(shape[1] * stretch_rate)

    # We can do shifting/stretching and cropping in a single affine
    # transform (including setting the upper bands to zero if we shift
    # down the signal so far that we exceed the input's nyquist rate)

    spec = spline_filter(spec.T, 2).astype(spec.dtype)
    spec = affine_transform(spec, (1 / stretch_rate, 1 / shift_rate),
                            output_shape=(new_length, shape[0]),
                            offset=(0, 0), mode='constant', order=2,
                            prefilter=False)

    # clip possible negative values introduced by the interpolation
    # delete the last frame if it is empty

    if (spec[-1] == 0).all():
        spec = spec[:-1]
    return np.maximum(spec.T, 0), stretch_rate
