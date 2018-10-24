import sys
sys.path.append("../..")
import glob
import os.path
import smd.utils as utils
import numpy as np

AUDIO_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/scheirer_slaney/audio"
FILELISTS_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/scheirer_slaney/filelists"


def load_files():
    audio_files = glob.glob(AUDIO_PATH + "/*.wav")
    return audio_files


if __name__ == "__main__":
    audio_files = load_files()

    print("Number of audio files: " + str(len(audio_files)))

    mixed = []
    music = []
    speech = []
    noise = []

    for file in audio_files:
        if "music" in os.path.basename(file):
            utils.save_annotation([["music"]], os.path.basename(file).replace(".wav", "") + ".txt", AUDIO_PATH)
            music.append(file)
        elif "speech" in os.path.basename(file):
            utils.save_annotation([["speech"]], os.path.basename(file).replace(".wav", "") + ".txt", AUDIO_PATH)
            speech.append(file)
        elif "noise" in os.path.basename(file):
            utils.save_annotation([["noise"]], os.path.basename(file).replace(".wav", "") + ".txt", AUDIO_PATH)
            noise.append(file)
        elif "mixed" in os.path.basename(file):
            events = [[0, 15, "speech"], [0, 15, "music"]]
            utils.save_annotation(events, os.path.basename(file).replace(".wav", "") + ".txt", AUDIO_PATH)
            mixed.append(file)

    music_train = np.random.choice(music, size=int(len(music) * 0.8), replace=False)
    speech_train = np.random.choice(speech, size=int(len(speech) * 0.8), replace=False)
    noise_train = np.random.choice(noise, size=int(len(noise) * 0.8), replace=False)
    mixed_train = np.random.choice(mixed, size=int(len(mixed) * 0.8), replace=False)

    for file in music:
        if file in music_train:
            with open(os.path.join(FILELISTS_PATH, 'music_train'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
        else:
            with open(os.path.join(FILELISTS_PATH, 'music_val'), 'a') as f:
                f.write(os.path.basename(file) + '\n')

    for file in speech:
        if file in speech_train:
            with open(os.path.join(FILELISTS_PATH, 'speech_train'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
        else:
            with open(os.path.join(FILELISTS_PATH, 'speech_val'), 'a') as f:
                f.write(os.path.basename(file) + '\n')

    for file in noise:
        if file in noise_train:
            with open(os.path.join(FILELISTS_PATH, 'noise_train'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
        else:
            with open(os.path.join(FILELISTS_PATH, 'noise_val'), 'a') as f:
                f.write(os.path.basename(file) + '\n')

    for file in mixed:
        if file in mixed_train:
            with open(os.path.join(FILELISTS_PATH, 'mixed_train'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
        else:
            with open(os.path.join(FILELISTS_PATH, 'mixed_val'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
