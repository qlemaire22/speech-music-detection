import preprocessing.config as config
import librosa as _librosa
from presets import Preset

librosa = Preset(_librosa)
librosa['sr'] = config.SAMPLING_RATE
librosa['hop_length'] = config.HOP_LENGTH
librosa['n_fft'] = config.FFT_WINDOW_SIZE


def log_melspectrogram(filename):
    y = librosa.load(filename)[0]
    S = librosa.feature.melspectrogram(y=y, n_mels=config.N_MELS, f_min=config.F_MIN, f_max=config.F_MAX)
    logS = librosa.power_to_db(S)
    return logS
