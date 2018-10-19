import sys
sys.path.append("../..")
import glob
import os.path
from smd.data.preprocessing import labels
from tqdm import tqdm
import scipy.io

DATASET_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/klaus_seyerlehner/"
AUDIO_PATH = DATASET_PATH
ANNOTATION_PATH = DATASET_PATH

# !!!!! This dataset is not labelled for speech, ... only for evaluation of music detection


def load_files():
    audio_files = glob.glob(AUDIO_PATH + "/*.mp3")
    label_files = glob.glob(ANNOTATION_PATH + "/*.label")
    return audio_files, label_files


def find_associated_label(audio_file, label_files):
    name = os.path.basename(audio_file).replace(".mp3", "")
    for label in label_files:
        if name in label:
            return label
    raise ValueError("Incorrect input: the associated label in not present.")


def get_event_list(label_file, type):
    events = []
    mat = scipy.io.loadmat(label_files[0])
    data = mat["data"]

    # Need to check the time betweek each data
    exit()
    return events


def concatenate_events(events_1, events_2):
    return events_1 + events_2


if __name__ == "__main__":
    audio_files, label_files = load_files()

    print("Number of audio files: " + str(len(audio_files)))
    print("Number of label files: " + str(len(label_files)))

    for audio in tqdm(audio_files):
        music = find_associated_label(audio, label_files)

        music_events = get_event_list(music, "music")

        labels.save_annotation(music_events, os.path.basename(audio).replace(".mp3", "") + ".txt", AUDIO_PATH)
