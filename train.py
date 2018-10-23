import argparse
from smd.data.data_generator import DataGenerator
from smd.data.dataset_loader import DatasetLoader
from smd.models.model_loader import load_model
from smd.data.data_augmentation import random_loudness_spec, random_filter_spec, block_mixing_spec, pitch_time_deformation_spec
from smd.data import preprocessing
import smd.utils as utils
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
import keras.models
import os


def training_data_processing(spec_file, annotation_file, mean, std, spec_file2=None, annotation_file2=None):
    spec = np.load(spec_file)

    # noise

    spec, stretching_rate = pitch_time_deformation_spec(spec)
    spec = random_filter_spec(spec)
    spec = random_loudness_spec(spec)
    label = preprocessing.get_label(
        annotation_file, spec.shape[1], stretching_rate=stretching_rate)

    if not(spec_file2 is None):
        spec2 = np.load(spec_file2)

        # noise

        spec2, stretching_rate2 = pitch_time_deformation_spec(spec2)
        spec2 = random_filter_spec(spec2)
        spec2 = random_loudness_spec(spec2)
        label2 = preprocessing.get_label(
            annotation_file2, spec2.shape[1], stretching_rate=stretching_rate2)
        spec, label = block_mixing_spec(spec, spec2, label, label2)
        exit()

    mels = preprocessing.get_scaled_mel_bands(spec)
    mels = preprocessing.normalize(mels, mean, std)
    return mels, label


def validation_data_processing(spec_file, annotation_file, mean, std):
    spec = np.load(spec_file)

    mels = preprocessing.get_scaled_mel_bands(spec)
    mels = preprocessing.normalize(mels, mean, std)
    n_frame = mels.shape[1]
    label = preprocessing.get_label(
        annotation_file, n_frame, stretching_rate=1)
    return mels, label


def train(train_set, val_set, cfg, config_name, resume, model_path):
    if resume:
        print("Loading the model to resume..")
        model = keras.models.load_model(model_path % config_name)
    else:
        print("Loading the network..")
        model = load_model(cfg["model"])

    if not(os.path.isdir("checkpoint")):
        os.makedirs("checkpoint")
        print("Checkpoint folder created.")

    csv_logger = CSVLogger('checkpoint/' + config_name +
                           '-training.log', append=resume)
    save_ckpt = ModelCheckpoint("checkpoint/weights." + config_name + ".hdf5", monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                period=1)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=3,
                                   verbose=0, mode='auto')

    lr_schedule = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='auto', min_lr=10e-7)

    callback_list = [save_ckpt, early_stopping, lr_schedule, csv_logger]

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

    parser.add_argument('--resume', type=bool, default=False,
                        help='set to true to restart a previous starning')

    parser.add_argument('--resume_model', type=str, default="checkpoint/weights.%(language)s.hdf5",
                        help='path of the model to load when the starting is resumed')

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

    train(train_set, val_set, cfg, args.config, args.resume, args.resume_model)
