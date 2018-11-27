import sys
sys.path.append("..")
import os
import glob
from tqdm import tqdm
from subprocess import Popen, PIPE
import smd.config as audio_config
import argparse
import smd.utils as utils
from shutil import copyfile


def remove_silence_files(dataset_folder, dataset):
    cfg = utils.load_json('../datasets.json')

    DATA_PATH = os.path.join(dataset_folder, cfg[dataset]["data_folder"])
    NEW_DATA_PATH = DATA_PATH + "_" + \
        str(audio_config.SAMPLING_RATE) + "_" + \
        str(audio_config.AUDIO_MAX_LENGTH) + "_" + \
        str(audio_config.N_MELS)

    if not(os.path.isdir(NEW_DATA_PATH)):
        raise ValueError(NEW_DATA_PATH + " does not exist.")

    audio_files = glob.glob(DATA_PATH + "/*.wav")

    for file in tqdm(audio_files):
        run_sox(file)


def run_sox(input_file):
    temp_file = input_file.replace('.wav', '_t.wav')
    command = "sox " + input_file + " " + temp_file + " silence -l 1 0.1 1% -1 0.1 1%"
    p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    output, err = p.communicate()
    copyfile(temp_file, input_file)
    os.remove(temp_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Remove the silence at the beginning and at the end of the files in a dataset that match the configuration of the system.")

    parser.add_argument('dataset', type=str,
                        help='name of the dataset to be processed')

    parser.add_argument('--data_location', type=str, default="/Users/quentin/Computer/DataSet/Music/speech_music_detection/",
                        help='the location of the data')

    args = parser.parse_args()

    remove_silence_files(args.data_location, args.dataset)
