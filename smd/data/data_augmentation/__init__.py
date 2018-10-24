"""Module for all the data augmentation functions."""

from __future__ import absolute_import

from .noise import add_random_noise
from .mixing import block_mixing_audio
from .mixing import block_mixing_spec
from .filter import random_filter_spec
from .pitch_time import pitch_shifting_audio
from .pitch_time import time_stretching_audio
from .pitch_time import pitch_time_deformation_spec
from .loudness import random_loudness_spec
from .loudness import random_loudness_audio

__all__ = ['add_random_noise',
           'block_mixing_audio',
           'block_mixing_spec',
           'random_filter_spec',
           'pitch_shifting_audio',
           'time_stretching_audio',
           'pitch_time_deformation_spec',
           'random_loudness_spec',
           'random_loudness_audio']
