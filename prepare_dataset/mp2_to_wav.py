import glob
import os
from shutil import copyfile
from subprocess import Popen, PIPE


AUDIO_PATH = "./sveriges_radio/audio"


def load_files():
    audio_files = glob.glob(AUDIO_PATH + "/*.WAV")
    return audio_files


def convert(input_file):
    new_name = input_file.replace(" ", "")
    os.rename(input_file, new_name)
    os.rename(input_file.replace(".wav", ".txt"), new_name.replace(".wav", ".txt"))
    input_file = new_name
    temp_file = input_file.replace('.WAV', '_t.WAV')
    command = "ffmpeg -i " + input_file + " " + temp_file
    p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    output, err = p.communicate()
    copyfile(temp_file, input_file)
    os.remove(temp_file)


if __name__ == '__main__':
    files = load_files()
    print("Number of file to convert: " + str(len(files)))
    for file in files:
        convert(file)
