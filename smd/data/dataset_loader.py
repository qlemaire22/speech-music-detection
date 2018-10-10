import json


class DatasetLoader():
    def __init__(self, dataset):
        with open('dataset.json') as f:
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
