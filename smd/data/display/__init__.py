"""Module for all the display functions."""

from __future__ import absolute_import

from .audio import display_waveform
from .audio import display_mel_bands
from .audio import display_spec

__all__ = ['display_waveform', 'display_spec', 'display_mel_bands']
