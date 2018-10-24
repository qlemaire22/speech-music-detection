import sys
sys.path.append("../..")
import glob
import os.path
import smd.utils as utils
from tqdm import tqdm
import csv
import numpy as np

DATASET_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/muspeak/"
AUDIO_PATH = os.path.join(DATASET_PATH, "audio")
FILELISTS_PATH = os.path.join(DATASET_PATH, "filelists")


def load_files():
    audio_files = glob.glob(AUDIO_PATH + "/*.mp3")
    label_files = glob.glob(AUDIO_PATH + "/*.csv")
    return audio_files, label_files


def find_associated_label(audio_file, label_files):
    name = os.path.basename(audio_file).replace(".mp3", "")
    for label in label_files:
        if name in label:
            return label
    raise ValueError("Incorrect input: the associated label in not present.")


def get_event_list(label_file):
    events = []
    lines = []

    with open(label_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            lines.append(row)

    for line in lines:
        if len(line) > 0:
            if line[2] == "s":
                type = "speech"
            elif line[2] == "m":
                type = "music"
            else:
                raise ValueError("Incorrect input: unknown type.")
            events.append([line[0], float(line[0]) + float(line[1]), type])

    return events


if __name__ == "__main__":
    audio_files, label_files = load_files()

    print("Number of audio files: " + str(len(audio_files)))
    print("Number of label files: " + str(len(label_files)))

    for audio in tqdm(audio_files):
        label = find_associated_label(audio, label_files)

        events = get_event_list(label)

        utils.save_annotation(events, os.path.basename(audio).replace(".mp3", "") + ".txt", AUDIO_PATH)

    train = np.random.choice(audio_files, size=int(len(audio_files) * 0.8), replace=False)
    np.random.shuffle(train)
    test = train[int(len(audio_files) * 0.7):]
    train = train[:int(len(audio_files) * 0.7)]

    for file in audio_files:
        if file in train:
            with open(os.path.join(FILELISTS_PATH, 'mixed_train'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
        elif file in test:
            with open(os.path.join(FILELISTS_PATH, 'mixed_test'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
        else:
            with open(os.path.join(FILELISTS_PATH, 'mixed_val'), 'a') as f:
                f.write(os.path.basename(file) + '\n')
