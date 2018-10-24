import sys
sys.path.append("../..")
import glob
import os.path
import smd.utils as utils
import numpy as np

MUSIC_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/gtzan/music_wav"
SPEECH_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/gtzan/speech_wav"
FILELISTS_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/gtzan/filelists"


def load_files():
    music_files = glob.glob(MUSIC_PATH + "/*.wav")
    speech_files = glob.glob(SPEECH_PATH + "/*.wav")
    return music_files, speech_files


if __name__ == "__main__":
    music_files, speech_files = load_files()

    print("Number of music files: " + str(len(music_files)))
    print("Number of speech files: " + str(len(speech_files)))

    for file in music_files:
        utils.save_annotation([["music"]], os.path.basename(file).replace(".wav", "") + ".txt", MUSIC_PATH)

    for file in speech_files:
        utils.save_annotation([["speech"]], os.path.basename(file).replace(".wav", "") + ".txt", SPEECH_PATH)

    music_train = np.random.choice(music_files, size=int(len(music_files) * 0.8), replace=False)
    speech_train = np.random.choice(speech_files, size=int(len(speech_files) * 0.8), replace=False)

    for file in music_files:
        if file in music_train:
            with open(os.path.join(FILELISTS_PATH, 'music_train'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
        else:
            with open(os.path.join(FILELISTS_PATH, 'music_val'), 'a') as f:
                f.write(os.path.basename(file) + '\n')

    for file in speech_files:
        if file in speech_train:
            with open(os.path.join(FILELISTS_PATH, 'speech_train'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
        else:
            with open(os.path.join(FILELISTS_PATH, 'speech_val'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
