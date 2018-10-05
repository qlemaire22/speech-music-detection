import matplotlib.pyplot as plt
import config
from presets import Preset
import librosa as _librosa
import librosa.display as _display
import os

AUDIO_FILE_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/GTZAN/music_wav/redhot.wav"
DURATION = 1
MONO = True

_librosa.display = _display

librosa = Preset(_librosa)
librosa['sr'] = config.SAMPLING_RATE
librosa['hop_length'] = config.HOP_LENGTH
librosa['n_fft'] = config.FFT_WINDOW_SIZE


def display_spectrogram():
    y = librosa.load(AUDIO_FILE_PATH, duration=DURATION, mono=MONO)[0]

    plt.subplot(3, 1, 2)
    librosa.display.waveplot(y)
    plt.title(os.path.basename(AUDIO_FILE_PATH))
    plt.show()

    S = librosa.feature.melspectrogram(y=y, n_mels=config.N_MELS, fmin=config.F_MIN, fmax=config.F_MAX)
    S = librosa.power_to_db(S)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-scale Mel spectrogram')
    plt.tight_layout()
    plt.show()

    duration = librosa.get_duration(y)
    print("Duration of the audio sample: " + repr(duration) + "s")
    print("New sampling rate : " + repr(config.SAMPLING_RATE))
    print("Number of frame : " + repr(S.shape[1]))
    print("Number of coefficient: ", config.N_MELS)
    print("Output shape: ", S.shape)


if __name__ == '__main__':
    display_spectrogram()
