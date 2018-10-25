import argparse
from smd.data.data_generator import DataGenerator
from smd.data.dataset_loader import DatasetLoader
from smd.data import preprocessing
import smd.utils as utils
import numpy as np
import keras.models
from tqdm import tqdm
import smd.evaluation
import os
# from multiprocessing import Pool
# from functools import partial


def test_data_processing(spec_file, annotation_file, mean, std):
    spec = np.load(spec_file)

    mels = preprocessing.get_scaled_mel_bands(spec)
    mels = preprocessing.normalize(mels, mean, std)
    n_frame = mels.shape[1]
    label = preprocessing.get_label(
        annotation_file, n_frame, stretching_rate=1)
    return mels, label


def evaluate(test_set, cfg, config_name, model_path, save_path):
    print("Loading the model to resume " + model_path + "..")
    if '%s' in model_path:
        model = keras.models.load_model(model_path % config_name)
    else:
        model = keras.models.load_model(model_path)
    print("Start the prediction..")

    predictions = []
    ground_truth = []

    for i in tqdm(range(test_set.__len__())):
        x, y = test_set.__getitem__(i)
        x = x.reshape((1, x.shape[0], x.shape[1]))
        predictions.append(model.predict(x, batch_size=1, verbose=0)[0].T)
        ground_truth.append(y.T)

    print("Post processing..")

    predictions_events = []
    ground_truth_events = []

    for p, gt in zip(predictions, ground_truth):
        predictions_events.append(preprocessing.label_to_annotation(np.around(p)))
        ground_truth_events.append(preprocessing.label_to_annotation(np.around(gt)))

    print("Evaluation..")
    # pool = Pool(processes=4)
    # func = partial(smd.evaluation.eval, ground_truth_events, predictions_events, segment_length=0.01, event_tolerance=0.5)
    # result = pool.map_async(func)
    result = smd.evaluation.eval(ground_truth_events, predictions_events, segment_length=0.01, event_tolerance=0.5)

    with open(os.path.join(save_path, "eval_" + config_name + ".txt"), 'w') as f:
        for item in result:
            f.write(item.__str__())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train a neural network for speech and music detection.")

    parser.add_argument('--config', type=str, default="test1",
                        help='the configuration of the training')

    parser.add_argument('--save_path', type=str, default=".",
                        help='the location where to save the evaluation')

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

    print("Creating the data generator..")

    test_set = DataGenerator(dataset.get_test_set(),
                             cfg["batch_size"],
                             cfg["target_seq_length"],
                             test_data_processing,
                             dataset.get_training_mean(),
                             dataset.get_training_std(),
                             set_type="test")

    evaluate(test_set, cfg, args.config, args.resume_model, args.save_path)
