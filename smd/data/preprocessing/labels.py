import numpy as np
from smd import config, utils


def get_label(filename, n_frame, stretching_rate=1):
    """Generate the label matrix of an audio sample based on its annotations."""
    events = utils.read_annotation(filename)
    label = np.zeros((2, n_frame), dtype=int)

    if events == [["speech"]]:
        label[0] = 1
    elif events == [["music"]]:
        label[1] = 1
    elif events == [["noise"]]:
        None
    else:
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

    return label


def label_to_annotation(label):
    """Return the formatted annotations based on the label matrix."""
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

    if t1_speech != -1:
        t2_speech = frame_to_time(len(label[0]))
        events.append([str(t1_speech), str(t2_speech), "speech"])
        t1_speech = -1
        t2_speech = -1

    if t1_music != -1:
        t2_music = frame_to_time(len(label[0]))
        events.append([str(t1_music), str(t2_music), "music"])
        t1_music = -1
        t2_music = -1

    return events


def label_to_annotation_extended(label):
    """Return the formatted annotations based on the label matrix."""
    t1_music = -1
    t1_speech = -1
    t2_music = -1
    t2_speech = -1
    t1_both = -1
    t1_none = -1
    t2_both = -1
    t2_none = -1

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

        if label[1][i] == 1 and label[0][i] == 1 and t1_both == -1:
            t1_both = frame_to_time(i)
        elif (label[1][i] == 0 or label[0][i] == 0) and t1_both != -1:
            t2_both = frame_to_time(i)
            events.append([str(t1_both), str(t2_both), "both"])
            t1_both = -1
            t2_both = -1

        if label[1][i] == 0 and label[0][i] == 0 and t1_none == -1:
            t1_none = frame_to_time(i)
        elif (label[1][i] == 1 or label[0][i] == 1) and t1_none != -1:
            t2_none = frame_to_time(i)
            events.append([str(t1_none), str(t2_none), "nothing"])
            t1_none = -1
            t2_none = -1

    if t1_speech != -1:
        t2_speech = frame_to_time(len(label[0]))
        events.append([str(t1_speech), str(t2_speech), "speech"])
        t1_speech = -1
        t2_speech = -1

    if t1_music != -1:
        t2_music = frame_to_time(len(label[0]))
        events.append([str(t1_music), str(t2_music), "music"])
        t1_music = -1
        t2_music = -1

    if t1_both != -1:
        t2_both = frame_to_time(len(label[0]))
        events.append([str(t1_both), str(t2_both), "both"])
        t1_both = -1
        t2_both = -1

    if t1_none != -1:
        t2_none = frame_to_time(len(label[0]))
        events.append([str(t1_none), str(t2_none), "nothing"])
        t1_none = -1
        t2_none = -1

    return events


def time_to_frame(time):
    """Return the number of the frame corresponding to a timestamp."""
    n_frame = round(time / config.HOP_LENGTH * config.SAMPLING_RATE)
    return int(n_frame)


def frame_to_time(n_frame):
    """Return the timestamp corresponding to a frame."""
    time = n_frame / config.SAMPLING_RATE * config.HOP_LENGTH
    return time
