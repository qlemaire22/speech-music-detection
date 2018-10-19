import sys
sys.path.append("../..")
import glob
import os.path
from smd.data.preprocessing import labels
from tqdm import tqdm
import xml.etree.ElementTree as ET
import dateparser
import datetime
import numpy as np

DATASET_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/sveriges_radio/"
AUDIO_PATH = DATASET_PATH
ANNOTATION_PATH = DATASET_PATH


def load_files():
    audio_files = glob.glob(AUDIO_PATH + "/*.WAV")
    label_files = glob.glob(ANNOTATION_PATH + "/*.XML")
    return audio_files, label_files


def find_associated_label(audio_file, label_files):
    name = os.path.basename(audio_file).replace(".WAV", "")
    for label in label_files:
        if name in label:
            return label
    raise ValueError("Incorrect input: the associated label in not present.")


def print_timedelta(time):
    s, ms = time.seconds, time.microseconds
    h = int(s / 3600.0)
    s = s - h * 3600
    m = int(s / 60.0)
    s = s - m * 60
    if m < 10:
        m = "0" + str(m)
    return str(h) + ":" + str(m) + ":" + str(s)
    # return str(s) + "." + str(ms)


def get_event_list(label_file):
    events = []
    file = open(label_file, encoding='utf-16-le')
    root = ET.fromstring(file.read())
    file.close()
    files = []
    events = []
    starting_time = None
    for child in root:
        if child.tag == "Track":
            for group in child:
                if group.tag == "Group":
                    for element in group:
                        if element.tag == "Element":
                            event = []
                            for sub_element in element:
                                # print(sub_element.tag)
                                if sub_element.tag == "File_Filename0":
                                    # print(sub_element.text)
                                    files.append(sub_element.text)
                                elif sub_element.tag == "Time_Start":
                                    time = dateparser.parse(
                                        sub_element.text).time()
                                    if starting_time is None:
                                        starting_time = time
                                    time = datetime.datetime.combine(datetime.date.today(
                                    ), time) - datetime.datetime.combine(datetime.date.today(), starting_time)
                                    event.append(print_timedelta(time))
                                elif sub_element.tag == "Time_Stop":
                                    time = dateparser.parse(
                                        sub_element.text).time()
                                    time = datetime.datetime.combine(datetime.date.today(
                                    ), time) - datetime.datetime.combine(datetime.date.today(), starting_time)
                                    event.append(print_timedelta(time))
                            events.append(event)

    print(len(files))
    print(len(events))
    for i in range(len(files)):
        print(files[i], events[i])
    print(len(files))
    print(len(events))
    print(len(np.unique(files)))
    print(len(np.unique(events)))
    return events


def concatenate_events(events_1, events_2):
    return events_1 + events_2


if __name__ == "__main__":
    audio_files, label_files = load_files()

    print("Number of audio files: " + str(len(audio_files)))
    print("Number of label files: " + str(len(label_files)))

    for audio in tqdm(audio_files):
        label = find_associated_label(audio, label_files)
        events = get_event_list(label)

        exit()

        labels.save_annotation(events, os.path.basename(
            audio).replace(".WAV", "") + ".txt", AUDIO_PATH)
