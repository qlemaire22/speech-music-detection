import os


def read_filelists(folder):
    files = ["mixed_val", "mixed_train", "mixed_test", "noise_train", "music_train", "music_val", "speech_train", "speech_val"]

    dic = {"mixed_val": [],
           "mixed_train": [],
           "mixed_test": [],
           "noise_train": [],
           "music_train": [],
           "music_val": [],
           "speech_train": [],
           "speech_val": []}

    for file in files:
        path = os.path.join(folder, file)
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                dic[file].append(line.replace('\n', ''))

    return dic
