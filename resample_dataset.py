import json
import os
import glob
from tqdm import tqdm
from subprocess import Popen, PIPE
import smd.data.preprocessing.config as audio_config
import re
import argparse
import csv
import smd.data.utils as utils
import smd.data.preprocessing.audio as preprocessing
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


def resample_dataset(dataset_folder, dataset):
    path = os.path.join(dir_path, 'smd/data/datasets.json')
    with open(path) as f:
        cfg = json.load(f)

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

            length, spec = librosa_analysis(new_file)
            n += length
            delta1 = spec - mean[:, None]
            mean += np.sum(delta1, axis=1) / n
            delta2 = spec - mean[:, None]
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

    events = []
    with open(annotation, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in spamreader:
            events.append(row)

    n = len(files)

    for i in range(n):
        t0 = i * audio_config.AUDIO_MAX_LENGTH
        t1 = (i + 1) * audio_config.AUDIO_MAX_LENGTH
        valid_events = []
        for event in events:
            e0 = float(event[0])
            e1 = float(event[1])
            if (t0 > e0 and t0 < e1 or (t1 > e0 and t1 < e1) or (t0 < e0 and t1 > e0)):
                new_t0 = max(0, e0 - t0)
                new_t1 = min(audio_config.AUDIO_MAX_LENGTH,
                             e1 - t0)
                valid_events.append([new_t0, new_t1, event[2]])

        new_annotation = files[i].replace(
            os.path.splitext(files[i])[1], '.txt')
        with open(new_annotation, 'w') as f:
            for event in valid_events:
                f.write(str(event[0]) + '\t' +
                        str(event[1]) + '\t' + event[2] + '\n')

    return files


def librosa_analysis(file):
    audio = preprocessing.load_audio(file)
    spec = preprocessing.get_log_melspectrogram(audio)

    length = spec.shape[1]

    return length, spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Resample a dataset to match the configuration of the system.")

    parser.add_argument('dataset', type=str,
                        help='name of the dataset to be resample')

    parser.add_argument('--data_location', type=str, default="/Users/quentin/Computer/DataSet/Music/speech_music_detection/",
                        help='the location of the data')

    args = parser.parse_args()

    resample_dataset(args.data_location, args.dataset)
