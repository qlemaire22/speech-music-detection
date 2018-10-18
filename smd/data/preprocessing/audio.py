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
    return librosa.load(filename, duration=duration)[0]


def log_melspectrogram(audio):
    S = librosa.feature.melspectrogram(y=audio, n_mels=config.N_MELS, fmin=config.F_MIN, fmax=config.F_MAX)
    logS = np.log1p(np.maximum(S, 0))
    return logS


def normalization(spec, mean, std):
    return (spec - mean) / std


def save_spec(spec, filename, dst):
    path = os.path.join(dst, filename)
    np.save(path, spec)
