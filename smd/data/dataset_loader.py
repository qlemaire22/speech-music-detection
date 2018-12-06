import os
import glob
import smd.utils as utils
import numpy as np
import smd.config as config


class DatasetLoader():
    def __init__(self, datasets, dataset_folder, datasets_config, verify=False):
        self.cfg = datasets_config
        self.verify = verify

        self.train_set = {"mixed": [],
                          "music": [],
                          "speech": [],
                          "noise": [],
                          "n_frame": 0,
                          "n_frame_mixed": 0,
                          "n_frame_speech": 0,
                          "n_frame_music": 0,
                          "n_frame_noise": 0}

        self.val_set = {"mixed": [],
                        "speech": [],
                        "music": [],
                        "noise": [],
                        "n_frame": 0}

        self.test_set = {"mixed": [],
                         "n_frame": 0}

        dataset_str = ""
        for dataset in datasets:
            dataset_str += "_" + dataset

        CACHED_MEAN_STD = False

        ns_val, ns_test, ns, ns_mixed, ns_speech, ns_music, ns_noise = [], [], [], [], [], [], []

        if os.path.isfile("checkpoint/mean" + dataset_str + ".npy") and os.path.isfile("checkpoint/std" + dataset_str + ".npy"):
            CACHED_MEAN_STD = True
            self.train_mean = np.load("checkpoint/mean" + dataset_str + ".npy")
            self.train_std = np.load("checkpoint/std" + dataset_str + ".npy")
        else:
            ms = []
            vs = []

        for dataset in datasets:
            suffix = '_' + str(config.SAMPLING_RATE) + '_' + str(config.AUDIO_MAX_LENGTH) + '_' + str(config.N_MELS)
            filelist_path = os.path.join(dataset_folder, self.cfg[dataset]["filelists_folder"] + suffix)
            data_path = os.path.join(dataset_folder, self.cfg[dataset]["data_folder"] + suffix)

            if not(os.path.exists(filelist_path)) or not(os.path.exists(data_path)):
                raise ValueError("The datatset " + dataset + " is unfound or has not been preprocessed for the chosen hyper-parameters.")

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
                elif "noise_val" in file:
                    self.load_list(file, "noise", self.val_set, data_path)
                elif "mixed_val" in file:
                    self.load_list(file, "mixed", self.val_set, data_path)
                elif "speech_val" in file:
                    self.load_list(file, "speech", self.val_set, data_path)
                elif "music_val" in file:
                    self.load_list(file, "music", self.val_set, data_path)
                elif "mixed_test" in file:
                    self.load_list(file, "mixed", self.test_set, data_path)
                elif "info.json" in file:
                    data = utils.load_json(file)
                    ns.append(data["N_FRAME_TRAIN"])
                    ns_val.append(data["N_FRAME_VAL"])
                    ns_test.append(data["N_FRAME_TEST"])
                    ns_mixed.append(data["N_FRAME_TRAIN_MIXED"])
                    ns_speech.append(data["N_FRAME_TRAIN_SPEECH"])
                    ns_music.append(data["N_FRAME_TRAIN_MUSIC"])
                    ns_noise.append(data["N_FRAME_TRAIN_NOISE"])
                elif "mean.npy" in file and not(CACHED_MEAN_STD):
                    ms.append(np.load(file))
                elif "var.npy" in file and not(CACHED_MEAN_STD):
                    vs.append(np.load(file))

        if not(CACHED_MEAN_STD):
            self.train_mean = self.combine_means(ms, ns)
            self.train_std = np.sqrt(self.combine_var(vs, ns, ms, self.train_mean))
            np.save("checkpoint/mean" + dataset_str + ".npy", self.train_mean)
            np.save("checkpoint/std" + dataset_str + ".npy", self.train_std)

        self.train_set["n_frame"] = np.sum(ns)
        self.val_set["n_frame"] = np.sum(ns_val)
        self.test_set["n_frame"] = np.sum(ns_test)
        self.train_set["n_frame_mixed"] = np.sum(ns_mixed)
        self.train_set["n_frame_speech"] = np.sum(ns_speech)
        self.train_set["n_frame_music"] = np.sum(ns_music)
        self.train_set["n_frame_noise"] = np.sum(ns_noise)

    def get_train_set(self):
        return self.train_set

    def get_val_set(self):
        return self.val_set

    def get_test_set(self):
        return self.test_set

    def get_training_mean(self):
        return self.train_mean

    def get_training_std(self):
        return self.train_std

    def load_list(self, list_file, label_type, data_list, data_path):
        with open(list_file, 'r') as f:
            files = f.readlines()
        for file in files:
            filename, length = file.replace('\n', '').split('\t')
            if self.verify:
                if not(os.path.isfile(os.path.join(data_path, filename))):
                    print("File not found: " + filename + " in " + list_file)
            data_list[label_type].append((os.path.join(data_path, filename), length))

    def combine_means(self, ms, ns):
        a, b = 0, 0
        for i in range(len(ms)):
            a += ns[i] * ms[i]
            b += ns[i]
        return a / b

    def combine_var(self, vs, ns, ms, mc):
        a, b = 0, 0
        for i in range(len(vs)):
            a += ns[i] * (vs[i] + (ms[i] - mc)**2)
            b += ns[i]
        return a / b
