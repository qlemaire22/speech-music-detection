import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from smd.models import tcn
import smd.config as config
import smd.utils as utils
from smd.data.data_generator import DataGenerator
from smd.data.dataset_loader import DatasetLoader
from smd.data.data_augmentation import random_loudness_spec, random_filter_spec, block_mixing_spec, pitch_time_deformation_spec
from smd.data import preprocessing

from keras import optimizers


def data():
    cfg = {"dataset": ["ofai"],
           "data_location": "/Users/quentin/Computer/DataSet/Music/speech_music_detection/",
           "target_seq_length": 270,
           "batch_size": 32
           }

    def training_data_processing(spec_file, annotation_file, mean, std, spec_file2=None, annotation_file2=None):
        spec = np.load(spec_file)
        spec, stretching_rate = pitch_time_deformation_spec(spec)
        spec = random_filter_spec(spec)
        spec = random_loudness_spec(spec)
        label = preprocessing.get_label(
            annotation_file, spec.shape[1], stretching_rate=stretching_rate)

        if not(spec_file2 is None):
            spec2 = np.load(spec_file2)
            spec2, stretching_rate2 = pitch_time_deformation_spec(spec2)
            spec2 = random_filter_spec(spec2)
            spec2 = random_loudness_spec(spec2)
            label2 = preprocessing.get_label(
                annotation_file2, spec2.shape[1], stretching_rate=stretching_rate2)
            spec, label = block_mixing_spec(spec, spec2, label, label2)

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

    datasets_config = utils.load_json("datasets.json")
    dataset = DatasetLoader(
        cfg["dataset"], cfg["data_location"], datasets_config)

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
    return train_set, val_set


def fit_tcn(train_set, val_set):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    cfg = {"optimizer":
           {
               "name": "SGD",
               "lr": 0.001,
               "momentum": 0.9,
               "decay": 0
           },
           "batch_size": 32,
           "workers": 8,
           "use_multiprocessing": True
           }

    model = tcn.create_tcn(nb_filters={{choice([16, 32, 64, 128])}},
                           kernel_size={{choice([3, 5, 7, 9])}},
                           dilations={{choice([[2 ** i for i in range(4)],
                                               [2 ** i for i in range(5)],
                                               [2 ** i for i in range(6)],
                                               [2 ** i for i in range(7)],
                                               [2 ** i for i in range(8)]])}},
                           nb_stacks={{choice([3, 4, 5, 6])}},
                           use_skip_connections={{choice([True, False])}},
                           dropout_rate={{uniform(0.05, 0.5)}})

    optimizer = optimizers.SGD(
        lr=cfg["optimizer"]["lr"], momentum=0.9, decay=cfg["optimizer"]["decay"])

    model.compile(loss=config.LOSS, metrics=config.METRICS,
                  optimizer=optimizer)

    result = model.fit_generator(train_set,
                                 epochs=5,
                                 validation_data=val_set,
                                 workers=cfg["workers"],
                                 use_multiprocessing=cfg["use_multiprocessing"],
                                 shuffle=True
                                 )
    # get the highest validation accuracy of the training epochs
    validation_loss = np.amin(result.history['val_loss'])
    print('Best validation acc of epoch:', validation_loss)
    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=fit_tcn,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
    print("Best found values:")
    print(best_run)
