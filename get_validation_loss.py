import argparse
from smd.data import preprocessing
from smd.data.dataset_loader import DatasetLoader
from smd.data.data_generator import DataGenerator
import smd.utils as utils
import numpy as np
import keras.models


def validation_data_processing(spec_file, annotation_file, mean, std):
    spec = np.load(spec_file)

    mels = preprocessing.get_scaled_mel_bands(spec)
    mels = preprocessing.normalize(mels, mean, std)
    n_frame = mels.shape[1]
    label = preprocessing.get_label(
        annotation_file, n_frame, stretching_rate=1)
    return mels, label


def get_validation_loss(val_set, model_path, cfg):
    print("Loading the model " + model_path + "..")
    model = keras.models.load_model(model_path)
    print("Start the prediction..")

    result = model.evaluate_generator(val_set,
                                      workers=cfg["workers"],
                                      use_multiprocessing=cfg["use_multiprocessing"],
                                      verbose=1)

    print("Final result:")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train a neural network for speech and music detection.")

    parser.add_argument('--config', type=str, default="high_quality",
                        help='the configuration of the training')

    parser.add_argument('--data_location', type=str, default="/Users/quentin/Computer/DataSet/Music/speech_music_detection/",
                        help='the location of the data')

    parser.add_argument('--model', type=str, default="trained/model.hdf5",
                        help='path of the model to load when the starting is resumed')

    parser.add_argument('--mean_path', type=str, default="trained/mean.npy",
                        help='path of the mean of the normalization applied with the model')

    parser.add_argument('--std_path', type=str, default="trained/std.npy",
                        help='path of the std of the normalization applied with the model')

    args = parser.parse_args()

    experiments = utils.load_json('experiments.json')
    cfg = experiments[args.config]

    print("Creating the dataset..")

    datasets_config = utils.load_json("datasets.json")
    dataset = DatasetLoader(
        cfg["dataset"], args.data_location, datasets_config)

    print("Creating the data generator..")

    val_set = DataGenerator(dataset.get_val_set(),
                            cfg["batch_size"],
                            cfg["target_seq_length"],
                            validation_data_processing,
                            dataset.get_training_mean(),
                            dataset.get_training_std(),
                            set_type="val")

    get_validation_loss(val_set, args.model, cfg)
