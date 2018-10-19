import sys
sys.path.append("..")
import json
import os
import glob
from tqdm import tqdm
from subprocess import Popen, PIPE
import smd.config as audio_config
import re
import argparse
import smd.utils as utils
import smd.data.preprocessing as preprocessing
import numpy as np


# save spec


def resample_dataset(dataset_folder, dataset):
    cfg = utils.load_json('../datasets.json')

    DATA_PATH = os.path.join(dataset_folder, cfg[dataset]["data_folder"])
    NEW_DATA_PATH = DATA_PATH + "_" + \
        str(audio_config.SAMPLING_RATE) + "_" + \
        str(audio_config.AUDIO_MAX_LENGTH) + "_" + \
        str(audio_config.N_MELS)

    FILELISTS_FOLDER = os.path.join(
        dataset_folder, cfg[dataset]["filelists_folder"])
    NEW_FILELISTS_FOLDER = FILELISTS_FOLDER + "_" + \
        str(audio_config.SAMPLING_RATE) + "_" + \
        str(audio_config.AUDIO_MAX_LENGTH) + "_" + \
        str(audio_config.N_MELS)

    if os.path.isdir(NEW_DATA_PATH):
        raise ValueError(NEW_DATA_PATH + " already exists.")
    else:
        os.makedirs(NEW_DATA_PATH)
        print("Output folder created: " + NEW_DATA_PATH)

    if os.path.isdir(NEW_FILELISTS_FOLDER):
        # raise ValueError(NEW_FILELISTS_FOLDER + " already exists.")
        None
    else:
        os.makedirs(NEW_FILELISTS_FOLDER)
        print("Output folder created: " + NEW_FILELISTS_FOLDER)

    audio_files = glob.glob(DATA_PATH + "/*.WAV")
    audio_files += glob.glob(DATA_PATH + "/*.wav")
    audio_files += glob.glob(DATA_PATH + "/*.mp3")

    new_audio_files = []

    filelists = utils.read_filelists(FILELISTS_FOLDER)

    n = 0
    mean = np.zeros(audio_config.N_MELS)
    std = np.zeros(audio_config.N_MELS)

    # specs = []

    for file in tqdm(audio_files[:2]):
        basename = os.path.basename(file)
        new_file = os.path.join(NEW_DATA_PATH, basename).replace(
            os.path.splitext(file)[1], '.wav')
        new_files = run_sox(file, new_file,
                            audio_config.SAMPLING_RATE, audio_config.AUDIO_MAX_LENGTH)
        new_audio_files += new_files

        for key in filelists.keys():
            if basename in filelists[key]:
                with open(os.path.join(NEW_FILELISTS_FOLDER, key), 'w') as f:
                    for new_file in new_files:
                        f.write(os.path.basename(new_file) + '\n')

        for new_file in new_files:
            for key in filelists.keys():
                if basename in filelists[key]:
                    with open(os.path.join(NEW_FILELISTS_FOLDER, key), 'w') as f:
                        f.write(os.path.basename(new_file) + '\n')

            length, bands = librosa_analysis(new_file)
            n += length
            delta1 = bands - mean[:, None]
            mean += np.sum(delta1, axis=1) / n
            delta2 = bands - mean[:, None]
            std += np.sum(delta1 * delta2, axis=1)

        # audio = preprocessing.load_audio(file)
        # specs.append(preprocessing.get_log_melspectrogram(audio))

    std /= (n - 1)
    std = np.sqrt(std)

    # spec = np.concatenate((specs[0], specs[1]), axis=1)
    # spec = preprocessing.normalize(spec, mean, std)

    # print(np.mean(np.mean(spec, axis=1)))
    # print(np.mean(np.std(spec, axis=1)))

    infos = {
        "TOTAL_LENGTH": n,
        "N_MELS": audio_config.N_MELS,
        "SAMPLING_RATE": audio_config.SAMPLING_RATE,
        "FFT_WINDOW_SIZE": audio_config.FFT_WINDOW_SIZE,
        "HOP_LENGTH": audio_config.HOP_LENGTH,
        "F_MIN": audio_config.F_MIN,
        "F_MAX": audio_config.F_MAX,
        "AUDIO_MAX_LENGTH": audio_config.AUDIO_MAX_LENGTH
    }
    np.save(os.path.join(NEW_FILELISTS_FOLDER, "bands_mean.npy"), mean)
    np.save(os.path.join(NEW_FILELISTS_FOLDER, "bands_std.npy"), std)
    with open(os.path.join(NEW_FILELISTS_FOLDER, "info.json"), 'w') as f:
        json.dump(infos, f)


def run_sox(input_file, output_file, sampling_rate, max_length):
    command = "sox -V3 -v 0.99 " + input_file + " -r " + str(sampling_rate) + \
        " -c 1 -b 32 " + output_file + " trim 0 " + \
        str(max_length) + " : newfile : restart"
    p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    output, err = p.communicate()
    data = err.decode("utf-8").split('\n')
    files = []
    for line in data:
        if "Output File" in line:
            files.append(re.search("'(.+?)'", line).group(1))

    annotation = input_file.replace(os.path.splitext(input_file)[1], '.txt')

    events = utils.read_annotation(annotation)

    n = len(files)

    for i in range(n):
        valid_events = []
        if events == [['music']]:
            valid_events = events
        elif events == [['speech']]:
            valid_events = events
        elif events == [['noise']]:
            valid_events = events
        else:
            t0 = i * audio_config.AUDIO_MAX_LENGTH
            t1 = (i + 1) * audio_config.AUDIO_MAX_LENGTH
            for event in events:
                e0 = float(event[0])
                e1 = float(event[1])
                if (t0 >= e0 and t0 < e1) or (t1 > e0 and t1 <= e1) or (t0 <= e0 and t1 >= e0):
                    new_t0 = max(0, e0 - t0)
                    new_t1 = min(audio_config.AUDIO_MAX_LENGTH,
                                 e1 - t0)
                    valid_events.append([new_t0, new_t1, event[2]])

        new_annotation = files[i].replace(
            os.path.splitext(files[i])[1], '.txt')

        utils.save_annotation(valid_events, new_annotation)

    return files


def librosa_analysis(file):
    audio = utils.load_audio(file)

    spec = preprocessing.get_spectrogram(audio)
    bands = preprocessing.get_scaled_mel_bands(spec)

    length = bands.shape[1]
    utils.save_matrix(spec, file.replace(".wav", ''))
    return length, bands


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Resample a dataset to match the configuration of the system.")

    parser.add_argument('dataset', type=str,
                        help='name of the dataset to be resample')

    parser.add_argument('--data_location', type=str, default="/Users/quentin/Computer/DataSet/Music/speech_music_detection/",
                        help='the location of the data')

    args = parser.parse_args()

    resample_dataset(args.data_location, args.dataset)
