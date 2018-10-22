import keras
import numpy as np
from math import ceil
import smd.config as config

"""
    - Only support mixed audio for now
    - verif batch construction ok
    verif dataset exist dataset model_loader
"""


class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, batch_size, target_seq_length, data_processing, mean, std, set_type='train'):
        self.dataset = dataset
        self.set_type = set_type
        self.batch_size = batch_size
        self.target_seq_length = target_seq_length
        self.data_processing = data_processing
        self.mean = mean
        self.std = std

        if set_type == 'train':
            self.length = dataset["n_frame_mixed"] + int((dataset["n_frame_music"] + dataset["n_frame_speech"]) / 2)
        elif set_type == 'val':
            self.length = dataset["n_frame"]

        self.nb_batch = ceil(self.length / (self.batch_size * self.target_seq_length))
        self.batch_composition = []

        self.on_epoch_end()

    def __getitem__(self, index):
        """ Return one batch and its corresponding label """
        X, Y = None, None
        for item in self.batch_composition[index]:
            mels, label = self.data_processing(item[0] + ".npy", item[0] + ".txt", self.mean, self.std)
            if X is None:
                X = mels.T
                Y = label.T
            else:
                X = np.concatenate((X, mels.T), axis=0)
                Y = np.concatenate((Y, label.T), axis=0)

        n_frame, dim = X.shape
        seq_length = ceil(n_frame / self.nb_batch)
        X.resize((self.nb_batch, seq_length, dim))
        Y.resize((self.nb_batch, seq_length, config.CLASSES))
        return X, Y

    def __len__(self):
        """ Return the number of batches """
        return self.nb_batch

    def on_epoch_end(self):
        """
            For non mixed possibility:
            create couples?
        """
        self.indexes = np.arange(len(self.dataset["mixed"]))
        np.random.shuffle(self.indexes)

        target_length = self.target_seq_length * self.batch_size

        self.batch_composition = []

        id = 0

        for i in range(self.nb_batch):
            sum = 0
            is_full = False
            self.batch_composition.append([])
            while not(is_full) and id < len(self.indexes):
                item = self.dataset["mixed"][self.indexes[id]]
                if sum + int(float(item[1]) * 0.70) <= target_length:
                    id += 1
                    self.batch_composition[i].append(item)
                    sum += int(item[1])
                else:
                    is_full = True
