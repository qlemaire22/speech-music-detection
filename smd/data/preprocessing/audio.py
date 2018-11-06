from smd import config
import librosa as _librosa
from presets import Preset
import numpy as np


librosa = Preset(_librosa)
librosa['sr'] = config.SAMPLING_RATE
librosa['hop_length'] = config.HOP_LENGTH
librosa['n_fft'] = config.FFT_WINDOW_SIZE


def get_spectrogram(audio):
    """Return the power magnitude spectrogram of the audio."""
    return np.abs(librosa.stft(audio))**2


def get_scaled_mel_bands(spec):
    """Return the log-scaled Mel bands of a power magnitude spectrogram"""
    filter = librosa.filters.mel(
        n_mels=config.N_MELS, fmin=config.F_MIN, fmax=config.F_MAX)
    bands = np.dot(filter, spec)
    return librosa.core.power_to_db(bands, amin=1e-7)


def normalize(bands, mean, std):
    """Normalize the Mel bands"""
    return (bands - mean[:, None]) / std[:, None]


def get_log_melspectrogram(audio):
    """Return the log-scaled Mel bands of an audio signal."""
    bands = librosa.feature.melspectrogram(
        y=audio, n_mels=config.N_MELS, fmin=config.F_MIN, fmax=config.F_MAX)
    return librosa.core.power_to_db(bands, amin=1e-7)
