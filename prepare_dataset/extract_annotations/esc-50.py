import sys
sys.path.append("../..")
import glob
import os.path
import smd.utils as utils
import numpy as np

NOISE_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/esc-50/audio"
FILELISTS_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/esc-50/filelists"


def load_files():
    noise_files = glob.glob(NOISE_PATH + "/*.wav")
    return noise_files


if __name__ == "__main__":
    noise_files = load_files()

    print("Number of noise files: " + str(len(noise_files)))

    for file in noise_files:
        utils.save_annotation([["noise"]], os.path.basename(file).replace(".wav", "") + ".txt", NOISE_PATH)

    noise_train = np.random.choice(noise_files, size=int(len(noise_files) * 0.8), replace=False)

    for file in noise_files:
        if file in noise_train:
            with open(os.path.join(FILELISTS_PATH, 'noise_train'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
        else:
            with open(os.path.join(FILELISTS_PATH, 'noise_val'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
