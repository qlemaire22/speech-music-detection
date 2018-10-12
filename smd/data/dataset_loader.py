import json
import os
import glob

dir_path = os.path.dirname(os.path.realpath(__file__))


class DatasetLoader():
    def __init__(self, datasets, dataset_folder):
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

        for dataset in datasets:
            filelist_path = os.path.join(dataset_folder, self.cfg[dataset]["filelists_folder"])
            data_path = os.path.join(dataset_folder, self.cfg[dataset]["data_folder"])

            files = glob.glob(filelist_path + "/*")

            for file in files:
                if "mixed_train" in file:
                    self.load_list(file, "mixed", self.train_set, data_path)
                elif "music_train" in file:
                    self.load_list(file, "music", self.train_set, data_path)
                elif "speech_train" in file:
                    self.load_list(file, "speech", self.train_set, data_path)
                elif "noise_train" in file:
                    self.load_list(file, "noise", self.train_set, data_path)
                elif "mixed_val" in file:
                    self.load_list(file, "mixed", self.val_set, data_path)
                elif "speech_val" in file:
                    self.load_list(file, "speech", self.val_set, data_path)
                elif "music_val" in file:
                    self.load_list(file, "music", self.val_set, data_path)
                elif "mixed_test" in file:
                    self.load_list(file, "mixed", self.test_set, data_path)

    def load_list(self, filename, label_type, data_list, data_path):
        with open(filename, 'r') as f:
            files = f.readlines()
        for file in files:
            data_list[label_type].append(os.path.join(data_path, file.replace('\n', '')))

    def get_train_set(self):
        return self.train_set

    def get_val_set(self):
        return self.val_set

    def get_test_set(self):
        return self.test_set
