import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.indices = np.arange(len(self.ids))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, index):
        """ Return one batch and its corresponding label """
        x, y = 0, 0
        return x, y

    def __len__(self):
        """ Return the number of batches """
        return 0

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def data_augmentation(filename):
        # load spec

        # get spec

        # transfo

        # get mel

        None
