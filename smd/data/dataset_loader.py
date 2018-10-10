import json
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


class DatasetLoader():
    def __init__(self, dataset):
        path = os.path.join(dir_path, 'datasets.json')
        with open(path) as f:
            self.cfg = json.load(f)

        self.train_set = {"mixed": [],
                          "music": [],
                          "speech": [],
                          "noise": []}

        self.val_set = {"mixed": [],
                        "speech": [],
                        "music": []}

        self.test_set = {"mixed": []}

    def get_train_set(self):
        return self.train_set

    def get_val_set(self):
        return self.val_set

    def get_test_set(self):
        return self.test_set
