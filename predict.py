import argparse
from smd.data.dataset_loader import DatasetLoader
from smd.data import preprocessing
import smd.utils as utils
import numpy as np
import keras.models
from tqdm import tqdm
import os
import glob


def test_data_processing(spec_file, mean, std):
    spec = np.load(spec_file)
    mels = preprocessing.get_scaled_mel_bands(spec)
    mels = preprocessing.normalize(mels, mean, std)
    return mels.T


def predict(data_path, dataset, cfg, config_name, model_path):
    print("Loading the model to resume " + model_path + "..")
    if '%s' in model_path:
        model = keras.models.load_model(model_path % config_name)
    else:
        model = keras.models.load_model(model_path)
    print("Start the prediction..")

    if os.path.isdir(data_path):
        files = glob.glob(os.path.abspath(data_path) + "/*.npy")
    else:
        files = [os.path.abspath(data_path)]

    for file in tqdm(files):
        x = test_data_processing(file, dataset.get_training_mean(), dataset.get_training_std())
        x = x.reshape((1, x.shape[0], x.shape[1]))
        output = model.predict(x, batch_size=1, verbose=0)[0].T
        annotation = preprocessing.label_to_annotation(np.around(output).astype(int))
        output_path = file.replace(".npy", '') + "_prediction.txt"
        utils.save_annotation(annotation, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train a neural network for speech and music detection.")

    parser.add_argument('--config', type=str, default="test1",
                        help='the configuration of the training')

    parser.add_argument('--data_path', type=str, default="audio_test/",
                        help='path to a file or a folder for prediction')

    parser.add_argument('--data_location', type=str, default="/Users/quentin/Computer/DataSet/Music/speech_music_detection/",
                        help='the location of the data')

    parser.add_argument('--resume_model', type=str, default="checkpoint/weights.%s.hdf5",
                        help='path of the model to load when the starting is resumed')

    args = parser.parse_args()

    experiments = utils.load_json('experiments.json')
    cfg = experiments[args.config]

    print("Creating the dataset..")

    datasets_config = utils.load_json("datasets.json")
    dataset = DatasetLoader(
        cfg["dataset"], args.data_location, datasets_config)

    predict(args.data_path, dataset, cfg, args.config, args.resume_model)
