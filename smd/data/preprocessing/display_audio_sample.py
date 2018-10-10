import matplotlib.pyplot as plt
import config
from presets import Preset
import librosa as _librosa
import librosa.display as _display
import os
import time

AUDIO_FILE_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/gtzan/music_wav/redhot.wav"
DURATION = None
MONO = True

_librosa.display = _display

librosa = Preset(_librosa)
librosa['sr'] = config.SAMPLING_RATE
librosa['hop_length'] = config.HOP_LENGTH
librosa['n_fft'] = config.FFT_WINDOW_SIZE


def display_spectrogram():
    t0 = time.time()
    y = librosa.load(AUDIO_FILE_PATH, duration=DURATION, mono=MONO)[0]
    t1 = time.time()

    plt.subplot(3, 1, 2)
    librosa.display.waveplot(y)
    plt.title(os.path.basename(AUDIO_FILE_PATH))
    plt.show()

    S_temp = librosa.core.stft(y)

    t2 = time.time()
    S = librosa.feature.melspectrogram(y=y, n_mels=config.N_MELS, fmin=config.F_MIN, fmax=config.F_MAX)
    S = librosa.power_to_db(S)
    t3 = time.time()

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-scale Mel spectrogram')
    plt.tight_layout()
    plt.show()

    duration = librosa.get_duration(y)
    print("Loading duration: " + repr(t1 - t0) + "s")
    print("Processing duration: " + repr(t3 - t2) + "s")
    print("Duration of the audio sample: " + repr(duration) + "s")
    print("New sampling rate : " + repr(config.SAMPLING_RATE))
    print("Number of frame : " + repr(S.shape[1]))
    print("Number of coefficient: ", config.N_MELS)
    print("Duration of a frame: " + repr(duration / S.shape[1]))
    print("Output shape: ", S.shape)
    print("STFT shape: ", S_temp.shape)


if __name__ == '__main__':
    display_spectrogram()
