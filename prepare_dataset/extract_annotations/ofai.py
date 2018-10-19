import sys
sys.path.append("../..")
import glob
import os.path
from smd.data.preprocessing import labels
from tqdm import tqdm

DATASET_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/ofai/"
AUDIO_PATH = os.path.join(DATASET_PATH, "audio")
MUSIC_ANNOTATION_PATH = os.path.join(DATASET_PATH, "labels/music")
SPEECH_ANNOTATION_PATH = os.path.join(DATASET_PATH, "labels/speech")


def load_files():
    audio_files = glob.glob(AUDIO_PATH + "/*.mp3")
    music_label_files = glob.glob(MUSIC_ANNOTATION_PATH + "/*.label")
    speech_label_files = glob.glob(SPEECH_ANNOTATION_PATH + "/*.label")
    return audio_files, music_label_files, speech_label_files


def find_associated_label(audio_file, label_files):
    name = os.path.basename(audio_file).replace(".mp3", "")
    for label in label_files:
        if name in label:
            return label
    raise ValueError("Incorrect input: the associated label in not present.")


def get_event_list(label_file, type):
    events = []
    with open(label_file, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.replace('\n', '').split('\t')
        if int(line[2]) == 1:
            events.append([line[0], float(line[0]) + float(line[1]), type])

    return events


def concatenate_events(events_1, events_2):
    return events_1 + events_2


if __name__ == "__main__":
    audio_files, music_label_files, speech_label_files = load_files()

    print("Number of audio files: " + str(len(audio_files)))
    print("Number of music label files: " + str(len(music_label_files)))
    print("Number of speech label files: " + str(len(speech_label_files)))

    for audio in tqdm(audio_files):
        music = find_associated_label(audio, music_label_files)
        speech = find_associated_label(audio, speech_label_files)

        music_events = get_event_list(music, "music")
        speech_events = get_event_list(speech, "speech")

        events = concatenate_events(music_events, speech_events)

        labels.save_annotation(events, os.path.basename(audio).replace(".mp3", "") + ".txt", AUDIO_PATH)
