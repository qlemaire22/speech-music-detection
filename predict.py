import argparse
from smd.data import preprocessing
import smd.utils as utils
import numpy as np
import keras.models
from tqdm import tqdm
import os
import glob


def test_data_processing(file, mean, std):
    if os.path.splitext(file)[1] == '.npy':
        spec = np.load(file)
    else:
        audio = utils.load_audio(file)
        spec = preprocessing.get_spectrogram(audio)
    mels = preprocessing.get_scaled_mel_bands(spec)
    mels = preprocessing.normalize(mels, mean, std)
    return mels.T


def predict(data_path, model_path, mean_path, std_path):
    mean = np.load(mean_path)
    std = np.load(std_path)

    print("Loading the model to resume " + model_path + "..")
    model = keras.models.load_model(model_path)
    print("Start the prediction..")

    if os.path.isdir(data_path):
        files = glob.glob(os.path.abspath(data_path) + "/*.npy") + glob.glob(os.path.abspath(data_path) + "/*.wav")
    else:
        files = [os.path.abspath(data_path)]

    for file in tqdm(files):
        x = test_data_processing(file, mean, std)
        x = x.reshape((1, x.shape[0], x.shape[1]))
        output = model.predict(x, batch_size=1, verbose=0)[0].T
        annotation = preprocessing.label_to_annotation(np.around(output).astype(int))
        output_path = file.replace(".npy", '') + "_prediction.txt"
        utils.save_annotation(annotation, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train a neural network for speech and music detection.")

    parser.add_argument('--data_path', type=str, default="audio_test/",
                        help='path to a file or a folder for prediction')

    parser.add_argument('--resume_model', type=str, default="trained/model.hdf5",
                        help='path of the model to load when the starting is resumed')

    parser.add_argument('--mean_path', type=str, default="trained/mean.npy",
                        help='path of the mean of the normalization applied with the model')

    parser.add_argument('--std_path', type=str, default="trained/std.npy",
                        help='path of the std of the normalization applied with the model')

    args = parser.parse_args()

    predict(args.data_path, args.resume_model, args.mean_path, args.std_path)
