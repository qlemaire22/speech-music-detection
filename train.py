import argparse
from smd.data.data_generator import DataGenerator
from smd.data.dataset_loader import DatasetLoader
from smd.models.model_loader import load_model
from smd.data import preprocessing
import smd.utils as utils
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
import os


"""
TODO:
    - possibility to start the training once again
    scheduler
"""


def training_data_processing(spec_file, annotation_file, mean, std):
    spec = np.load(spec_file)

    # noise

    # stretching, shifting

    # random filters

    # random loudness

    mels = preprocessing.get_scaled_mel_bands(spec)
    mels = preprocessing.normalize(mels, mean, std)
    n_frame = mels.shape[1]
    label = preprocessing.get_label(
        annotation_file, n_frame, stretching_rate=1)
    return mels, label


def validation_data_processing(spec_file, annotation_file, mean, std):
    spec = np.load(spec_file)

    mels = preprocessing.get_scaled_mel_bands(spec)
    mels = preprocessing.normalize(mels, mean, std)
    n_frame = mels.shape[1]
    label = preprocessing.get_label(
        annotation_file, n_frame, stretching_rate=1)
    return mels, label


def train(train_set, val_set, cfg, config_name):
    print("Loading the network..")

    model = load_model(cfg["model"])

    if not(os.path.isdir("checkpoint")):
        os.makedirs("checkpoint")
        print("Checkpoint folder created.")

    csv_logger = CSVLogger('checkpoint/' + config_name + '-training.log', append=False)
    save_ckpt = ModelCheckpoint("checkpoint/weights." + config_name + ".hdf5", monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                period=1)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=2,
                                   verbose=0, mode='auto')

    # lr_schedule = LearningRateScheduler

    callback_list = [save_ckpt, csv_logger, early_stopping]

    print("Start the training..")

    model.fit_generator(train_set,
                        epochs=cfg["nb_epoch"],
                        callbacks=callback_list,
                        validation_data=val_set,
                        workers=cfg["workers"],
                        use_multiprocessing=cfg["use_multiprocessing"],
                        shuffle=True
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train a neural network for speech and music detection.")

    parser.add_argument('--config', type=str, default="test1",
                        help='the configuration of the training')

    parser.add_argument('--data_location', type=str, default="/Users/quentin/Computer/DataSet/Music/speech_music_detection/",
                        help='the location of the data')

    args = parser.parse_args()

    experiments = utils.load_json('experiments.json')
    cfg = experiments[args.config]

    print("Creating the dataset..")

    datasets_config = utils.load_json("datasets.json")
    dataset = DatasetLoader(
        cfg["dataset"], args.data_location, datasets_config)

    print("Creating the data generator..")

    train_set = DataGenerator(dataset.get_train_set(),
                              cfg["batch_size"],
                              cfg["target_seq_length"],
                              training_data_processing,
                              dataset.get_training_mean(),
                              dataset.get_training_std(),
                              set_type="train")

    val_set = DataGenerator(dataset.get_val_set(),
                            cfg["batch_size"],
                            cfg["target_seq_length"],
                            validation_data_processing,
                            dataset.get_training_mean(),
                            dataset.get_training_std(),
                            set_type="val")

    train(train_set, val_set, cfg, args.config)
