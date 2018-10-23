import smd.config as config
import numpy as np


def random_filter_spec(spec, min_std=5, max_std=7):
    """ Based on https://github.com/f0k/ismir2018/blob/master/experiments/augment.py """
    coef, length = spec.shape

    # sample means and std deviations on logarithmic pitch scale

    min_pitch = 12 * np.log2(150)
    max_pitch = 12 * np.log2(config.F_MAX)
    mean = min_pitch + (np.random.rand() * (max_pitch - min_pitch))
    std = min_std + np.random.randn() * (max_std - min_std)

    # convert means and std deviations to linear frequency scale

    std = 2**((mean + std) / 12) - 2**(mean / 12)
    mean = 2**(mean / 12)

    # convert means and std deviations to bins

    mean = mean * coef / config.F_MAX
    std = std * coef / config.F_MAX

    # sample strengths uniformly in dB

    strength = config.MAX_LOUDNESS_DB * 2 * (np.random.rand() - .5)

    # create Gaussians

    filt = (strength * np.exp(np.square((np.arange(coef) - mean) / std) * -.5))

    # transform from dB to factors

    filt = 10**(filt / 20.)

    # apply

    filt = np.asarray(filt, dtype=spec.dtype)
    return spec * filt[:, None]
