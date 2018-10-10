import argparse
from smd.data.data_generator import DataGenerator
from smd.data.dataset_loader import DatasetLoader
from smd.models.model_loader import load_model
import json


def train(train_set, val_set, cfg):
    model = load_model(cfg["model"])
    
    model.fit_generator(train_set,
                        epochs=cfg["nb_epoch"],
                        callbacks=None,
                        validation_data=val_set,
                        workers=cfg["workers"],
                        use_multiprocessing=cfg["se_multiprocessing"],
                        shuffle=True
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train a neural network for speech and music detection.")

    parser.add_argument('--config', type=str, default="test1",
                        help='the configuration of the training')

    parser.add_argument('--data_location', type=str, default="/Users/quentin/Computer/DataSet/Music/speech_music_detection/",
                        help='the locaftion of the data')

    args = parser.parse_args()

    with open('config.json') as f:
        data = json.load(f)
        cfg = data[args.config]

    dataset = DatasetLoader(cfg["dataset"])

    train_set = DataGenerator(dataset.get_train_set(), cfg["batch_size"])
    val_set = DataGenerator(dataset.get_val_set(), cfg["batch_size"])
    # BATCH SIZE VAL ?

    train(train_set, val_set, cfg)
