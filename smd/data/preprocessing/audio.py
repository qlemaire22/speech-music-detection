from smd.data.preprocessing import config
import librosa as _librosa
from presets import Preset
import os
import numpy as np


librosa = Preset(_librosa)
librosa['sr'] = config.SAMPLING_RATE
librosa['hop_length'] = config.HOP_LENGTH
librosa['n_fft'] = config.FFT_WINDOW_SIZE


def load_audio(filename, duration=None):
    """Load the audio file into a numpy array."""
    return librosa.load(filename, duration=duration)[0]


def get_spectrogram(audio):
    """Return the power magnitude spectrogram of the audio."""
    return np.abs(librosa.stft(audio))**2


def get_scaled_mel_bands(spec):
    """Return the log-scaled Mel bands of a power magnitude spectrogram"""
    bands = librosa.filters.mel(
        n_mels=config.N_MELS, fmin=config.F_MIN, fmax=config.F_MAX)
    return librosa.core.power_to_db(bands, amin=1e-7)


def normalize(bands, mean, std):
    """Normalize the Mel bands"""
    return (bands - mean[:, None]) / std[:, None]


def save_spec(spec, filename, dst):
    """Save a spectrogram into a .npy file"""
    path = os.path.join(dst, filename)
    np.save(path, spec)


def get_log_melspectrogram(audio):
    """Return the log-scaled Mel bands of an audio signal."""
    bands = librosa.feature.melspectrogram(
        y=audio, n_mels=config.N_MELS, fmin=config.F_MIN, fmax=config.F_MAX)
    return librosa.core.power_to_db(bands, amin=1e-7)
