import sys
sys.path.append("..")
from smd.data.dataset_loader import DatasetLoader
import argparse
from smd import utils
from tqdm import tqdm
import numpy as np


def get_statistics(dataset):
    set = dataset.get_train_set()["mixed"]

    music_events = []
    music_breaks = []
    speech_events = []
    speech_breaks = []

    for file in tqdm(set):
        last_music = None
        last_speech = None

        annotation = utils.read_annotation(file[0] + ".txt")
        annotation = sorted(annotation, key=lambda x: x[0])
        for event in annotation:
            event = (float(event[0]), float(event[1]), event[2])
            if event[2] == "speech":
                speech_events.append(event[1] - event[0])
                if not(last_speech is None):
                    speech_breaks.append(event[0] - last_speech)
                last_speech = event[1]
            else:
                music_events.append(event[1] - event[0])
                if not(last_music is None):
                    if event[0] - last_music > 0:
                        music_breaks.append(event[0] - last_music)
                last_music = event[1]

    np.save("speech_events.npy", speech_events)
    np.save("speech_breaks.npy", speech_breaks)
    np.save("music_events.npy", music_events)
    np.save("music_breaks.npy", music_breaks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to get some statistics on data.")

    parser.add_argument('--config', type=str, default="test1",
                        help='the configuration of the training')

    parser.add_argument('--data_location', type=str, default="/Users/quentin/Computer/DataSet/Music/speech_music_detection/",
                        help='the location of the data')

    args = parser.parse_args()

    experiments = utils.load_json('../experiments.json')
    cfg = experiments[args.config]

    print("Creating the dataset..")

    datasets_config = utils.load_json("../datasets.json")
    dataset = DatasetLoader(
        cfg["dataset"], args.data_location, datasets_config)

    get_statistics(dataset)
