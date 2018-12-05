import sys
sys.path.append("../..")
import glob
import os.path
import smd.utils as utils
import numpy as np
from subprocess import Popen, PIPE
from shutil import copyfile

AUDIO_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/sveriges_radio/audio"
FILELISTS_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/sveriges_radio/filelists"


def load_files():
    audio_files = glob.glob(AUDIO_PATH + "/*.wav")
    return audio_files


def remove_beginning(input_file):
    temp_file = input_file.replace('.wav', '_t.wav')
    command = "sox " + input_file + " " + temp_file + " trim 3"
    p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    output, err = p.communicate()
    copyfile(temp_file, input_file)
    os.remove(temp_file)


if __name__ == "__main__":
    audio_files = load_files()

    print("Number of audio files: " + str(len(audio_files)))

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
            if "totrim" in os.path.basename(file):
                remove_beginning(file)
        elif "noise" in os.path.basename(file):
            utils.save_annotation([["noise"]], os.path.basename(file).replace(".wav", "") + ".txt", AUDIO_PATH)
            noise.append(file)

    music_train = np.random.choice(music, size=int(len(music) * 0.8), replace=False)
    speech_train = np.random.choice(speech, size=int(len(speech) * 0.8), replace=False)
    noise_train = np.random.choice(noise, size=int(len(noise) * 0.8), replace=False)
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
