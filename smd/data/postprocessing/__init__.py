"""Module for audio postprocessing"""

from __future__ import absolute_import

from .smoothing import smooth_output
from .threshold import apply_threshold

__all__ = ['smooth_output',
           'apply_threshold']
