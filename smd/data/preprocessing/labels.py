import numpy as np
import csv
from smd.data.preprocessing import config
import os


def get_label(type, n_frame, filename=None, stretching_rate=1):
    r"""Generate the formatted label of an audio sample.

    Keyword arguments:

    type: speech, music, noise or mixed
    """
    label = np.zeros((2, n_frame), dtype=int)

    if type == "speech":
        label[0] = 1
    elif type == "music":
        label[1] = 1
    elif type == "noise":
        None
    elif type == "mixed":
        if filename is None:
            raise ValueError("If type=mixed, then filename must be set.")
        else:
            events = []
            with open(filename, 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
                for row in spamreader:
                    events.append(row)
            for event in events:
                f1 = time_to_frame(float(event[0]) * stretching_rate)
                f2 = time_to_frame(float(event[1]) * stretching_rate)
                if f2 < f1:
                    raise ValueError("An error occured in the annotation file " + filename + ", f1 > f2.")

                if event[2] == 'speech':
                    label[0][f1:f2] = 1
                elif event[2] == 'music':
                    label[1][f1:f2] = 1
                else:
                    raise ValueError("An error occured in the annotation file " + filename + ", unknown type.")
    else:
        raise ValueError("type must me either speech, music, noise or mixed.")

    return label


def get_annotation(label):
    t1_music = -1
    t1_speech = -1
    t2_music = -1
    t2_speech = -1

    events = []

    for i in range(len(label[0])):
        if label[0][i] == 1 and t1_speech == -1:
            t1_speech = frame_to_time(i)
        elif label[0][i] == 0 and t1_speech != -1:
            t2_speech = frame_to_time(i)
            events.append([str(t1_speech), str(t2_speech), "speech"])
            t1_speech = -1
            t2_speech = -1

        if label[1][i] == 1 and t1_music == -1:
            t1_music = frame_to_time(i)
        elif label[1][i] == 0 and t1_music != -1:
            t2_music = frame_to_time(i)
            events.append([str(t1_music), str(t2_music), "music"])
            t1_music = -1
            t2_music = -1

    return events


def time_to_frame(time):
    n_frame = round(time / config.HOP_LENGTH * config.SAMPLING_RATE)
    return n_frame


def frame_to_time(n_frame):
    time = n_frame / config.SAMPLING_RATE * config.HOP_LENGTH
    return time


def save_label(label, filename, dst):
    path = os.path.join(dst, filename)
    np.save(path, label)


def save_annotation(events, filename, dst):
    path = os.path.join(dst, filename)
    with open(path, "w") as f:
        for event in events:
            f.write(str(event[0]) + '\t' + str(event[1]) + '\t' + event[2] + '\n')
