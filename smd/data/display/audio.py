import matplotlib.pyplot as plt
import librosa as _librosa
import librosa.display as _display
from smd import config
from presets import Preset
import numpy as np

librosa = Preset(_librosa)
librosa['sr'] = config.SAMPLING_RATE
librosa['hop_length'] = config.HOP_LENGTH
librosa['n_fft'] = config.FFT_WINDOW_SIZE
_librosa.display = _display


def display_waveform(audio, figname="Waveform", figsize=None):
    plt.figure(figsize=figsize)
    librosa.display.waveplot(audio)
    plt.title(figname)
    plt.show()


def display_mel_bands(bands, figsize=None):
    plt.figure(figsize=figsize)
    librosa.display.specshow(bands, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-scale Mel spectrogram')
    plt.tight_layout()
    plt.show()


def display_spec(spec, figsize=None):
    plt.figure(figsize=figsize)
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.tight_layout()
    plt.show()
