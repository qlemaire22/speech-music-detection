import keras
import numpy as np
from math import ceil
import smd.config as config
import warnings
import random


class DataGenerator(keras.utils.Sequence):
    """ Data genertor class for speech and music detection."""

    def __init__(self, dataset, batch_size, target_seq_length, data_processing, mean, std, set_type='train'):
        self.dataset = dataset
        self.set_type = set_type
        self.batch_size = batch_size
        self.target_seq_length = target_seq_length
        self.data_processing = data_processing
        self.mean = mean
        self.std = std

        if set_type == 'train':
            self.length = dataset["n_frame_mixed"] + dataset["n_frame_noise"] + int(
                (dataset["n_frame_music"] + dataset["n_frame_speech"]) * (config.BLOCK_MIXING_MIN + config.BLOCK_MIXING_MAX) / 2)
            self.nb_batch = ceil(
                self.length / (self.batch_size * self.target_seq_length))
        elif set_type == 'val':
            self.length = dataset["n_frame"]
            self.nb_batch = ceil(
                self.length / (self.batch_size * self.target_seq_length))
        elif set_type == 'test':
            self.length = dataset["n_frame"]
            self.nb_batch = len(dataset["mixed"])

        self.batch_composition = []

        print("Batch size: " + str(self.batch_size))
        print("Number batches: " + str(self.nb_batch))
        print("Target seq_length: " + str(self.target_seq_length))

        self.on_epoch_end()

    def __getitem__(self, index):
        """ Return one batch and its corresponding label """
        X, Y = None, None
        for item in self.batch_composition[index]:
            if len(item) == 2:
                mels, label = self.data_processing(
                    item[0] + ".npy", item[0] + ".txt", self.mean, self.std)
            else:
                mels, label = self.data_processing(
                    item[1][0] + ".npy", item[1][0] + ".txt", self.mean, self.std,
                    spec_file2=item[2][0] + ".npy",
                    annotation_file2=item[2][0] + ".txt")
            if X is None:
                X = mels.T
                Y = label.T
            else:
                X = np.concatenate((X, mels.T), axis=0)
                Y = np.concatenate((Y, label.T), axis=0)

        if not(self.set_type == "test"):
            n_frame, dim = X.shape
            seq_length = ceil(n_frame / self.batch_size)
            return np.resize(X, (self.batch_size, seq_length, dim)), np.resize(Y, (self.batch_size, seq_length, config.CLASSES))
        else:
            return X, Y

    def __len__(self):
        """ Return the number of batches """
        return self.nb_batch

    def on_epoch_end(self):
        self.batch_composition = []

        if self.set_type == "test":
            for item in self.dataset["mixed"]:
                self.batch_composition.append([item])
            return
        elif self.set_type == "train":
            self.indexes = ["1_" + str(i)
                            for i in range(len(self.dataset["mixed"]))]
            self.indexes += ["5_" + str(i)
                             for i in range(len(self.dataset["noise"]))]

            music = list(range(len(self.dataset["music"])))
            speech = list(range(len(self.dataset["speech"])))
            random.shuffle(music)
            random.shuffle(speech)
            index = 0
            music_speech = []
            while index < len(music) and index < len(speech):
                music_speech.append("2_" + str(speech[index]) + '_' + str(music[index]))
                index += 1

            self.indexes += music_speech

            if index < len(speech):
                self.indexes += ["3_" + str(i)
                                 for i in range(index, len(speech))]
            elif index < len(music):
                self.indexes += ["4_" + str(i)
                                 for i in range(index, len(music))]
        elif self.set_type == "val":
            self.indexes = ["1_" + str(i)
                            for i in range(len(self.dataset["mixed"]))]
            self.indexes += ["3_" + str(i)
                             for i in range(len(self.dataset["speech"]))]
            self.indexes += ["4_" + str(i)
                             for i in range(len(self.dataset["music"]))]
            self.indexes += ["5_" + str(i)
                             for i in range(len(self.dataset["noise"]))]

        np.random.shuffle(self.indexes)

        target_length = self.target_seq_length * self.batch_size

        id = 0

        for i in range(self.nb_batch):
            sum = 0
            is_full = False
            self.batch_composition.append([])
            while not(is_full) and id < len(self.indexes):
                info = self.indexes[id].split('_')
                if info[0] == "1":
                    item = self.dataset["mixed"][int(info[1])]
                    length = item[1]
                elif info[0] == "2":
                    item1 = self.dataset["speech"][int(info[1])]
                    item2 = self.dataset["music"][int(info[2])]
                    length = (float(item1[1]) + float(item2[1])) * \
                        (config.BLOCK_MIXING_MIN + config.BLOCK_MIXING_MAX) / 2
                    item = [None, item1, item2]
                elif info[0] == "3":
                    item = self.dataset["speech"][int(info[1])]
                    length = item[1]
                elif info[0] == "4":
                    item = self.dataset["music"][int(info[1])]
                    length = item[1]
                elif info[0] == "5":
                    item = self.dataset["noise"][int(info[1])]
                    length = item[1]

                if sum + int(float(length) * 0.60) <= target_length:
                    id += 1
                    self.batch_composition[i].append(item)
                    sum += int(float(length))
                else:
                    is_full = True

        empty = 0
        for i in range(len(self.batch_composition)):
            if self.batch_composition[i] == []:
                empty += 1

        if empty > 0:
            warnings.warn("Some of the batches are empty: " + str(empty))
            warnings.warn(
                "Please consider reducing the max_length of the audio files or increasing the target_length.")

        if id < len(self.indexes) - 1:
            warnings.warn(
                "Some audio files could not enter in the batch composition. Excluded files: " + str(len(self.indexes) - 1 - id))
            warnings.warn(
                "Please consider reducing the max_length of the audio files.")
