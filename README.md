# Speech and Music Detection

Python framework for Speech and Music Detection using Keras.

This repository contains the experiments presented in the paper "Temporal Convolutional Networks for Speech and Music Detection in Radio Broadcast" by Quentin Lemaire and Andre Holzapfel at the 20th International Society for Music Information Retrieval conference (ISMIR 2019). [Paper](http://archives.ismir.net/ismir2019/paper/000026.pdf)

## Description

This framework is designed to easily evaluate new models and configurations for the speech and music detection task using neural networks. More details about this task can be found in the description page for the [MIREX 2018 Speech/Music Detection task](https://www.music-ir.org/mirex/wiki/2018:Music_and/or_Speech_Detection). The evaluation implemented in this framework is the same as the one described in this page for comparison purposes.

Different data pre-processing, data augmentation and architectures are already implemented and it is possible to easily add new methods and to train on different datasets.

## Installation

The SoX command line utility is required for the dataset pre-processing. ([HomePage](http://sox.sourceforge.net)).

Installation with HomeBrew:

    brew install lame
    brew reinstall sox --with-lame  # for mp3 compatibility

The implementation is based on several Python libraries, especially:

- `Keras` for the deep learning implementations [(link)](https://github.com/keras-team/keras).
- `tensorflow` as Keras backend [(link)](https://www.tensorflow.org).
- `Librosa` for the pre-processing of the audio [(link)](https://github.com/librosa/librosa).
- `sed_eval` for the evaluation of the models [(link)](https://github.com/TUT-ARG/sed_eval).
- `keras-tcn` for the implementation of the TCN [(link)](https://github.com/philipperemy/keras-tcn).
- `hyperas` for hyper-parameters optimization on Keras with Hyperopt [(link)](https://github.com/maxpumperla/hyperas).

If you want to use `prepare_dataset/mp2_to_wav.py` to convert MP2 audio to WAV, you need the command line utility `FFmpeg` ([HomePage](https://www.ffmpeg.org)).

Installation of the Python libraries with PyPi:

    pip install -r requirements.txt

To listen to the audio while visualizing the annotations with `smd.display.audio_with_events`, you need the toolbox `sed_vis` that you can download on [GitHub](https://github.com/TUT-ARG/sed_vis).

## Configuration

The different parameters of the framework that are not supposed to be changed when comparing different architectures can be set in `smd/config.py`.

Those parameters are:

- The preprocessing parameters like the sampling rate.
- The data augmentation parameters.
- The loss and metric used for the training.

## Data

### Labels

The label file of an audio can either be a text file containing one label for the whole file (speech, music or noise) or be a text file containing the list of the events happening in the audio. In the last case the audio will be considered "mixed" and the label file has to be formatted in this way:

    t_start1 t_stop1 music/speech
    t_start2 t_stop2 music/speech
    ...

Each value is separated by a tabulation.

### Add a dataset

The dataset has to be separated into two folders:

- The folder containing all the audio files and their corresponding label text files.
- The folder containing the repartition of the data between each set (train, validation or test) for each type of label (speech, music, noise or mixed). The files contain the name of the corresponding audio with no extension and the possible files are `mixed_train, mixed_val, mixed_test, music_train, music_val, speech_train, speech_val` or `noise_train`.

Then, add the name of the two folders in `datasets.json` and run the file `prepare_dataset/prepare_audio.py` to do the processing pre-training of the audio.

### Pre-processing

Prior to the learning phase, each audio file is resampled to a 22.05kHz mono audio of maximum 1mn30 (the files are split). Then a Short-Time Fourier Transform (STFT) with a Hann window, a frame length of 1024 and Hop size of 512 is applied and only the magnitude of the power spectrogram is kept. Those matrices are then stored for the learning phase. Those steps are done in the file `prepare_dataset/prepare_audio.py`.

During the learning phase, the spectrograms are loaded, then deformed by the data augmentation and a Mel Filterbank with 80 coefficients between 27.5 and 8000 Hz is applied. Those coefficients are then put on a log scale, normalized over the training set and inputted into the network. Those steps are implemented in `train.py` and could be changed to test new configurations.

### Data Augmentation

Different transformations are applied to each training sample to do some data augmentation. To reduce the computation during the training, the data augmentation is not applied to the audio signal but on the magnitude of the power spectrogram. The implemented transformations are:

- Time stretching
- Pitch shifting
- Random loudness
- Block mixing
- Gaussian frequency filter

Those deformations and the hyper-parameters used are based on the work of J. Schlüter and T. Grill [1].

## Experiment configuration

You can configure the model and the parameters of the training in the file `experiments.json`. Different models and optimizers can be chosen for this task.

The configuration for an experiment must have the following fields:

    "experiment_name": {
      "model": {
        ...
      },
      "dataset": [
        "list of the datasets to use"
      ],
      "batch_size": 32,
      "nb_epoch": 10,
      "target_seq_length": 270,
      "workers": 8,
      "use_multiprocessing": true
    }

### Models

New models can be added in the folder `smd/models` and loaded in `smd/models/model_loader.py`.

Here are the architectures already implemented with their configuration:

#### LTSM

    "model": {
      "type": "lstm",
      "hidden_units": [100, 100, 100],
      "dropout": 0.2,
      "bidirectional": true,
      "optimizer": ...
    }

#### CLDNN (Convolutional, Long Short-Term Memory, Fully Connected Deep Neural Networks)

    "model": {
      "type": "cldnn",
      "filters_list": [32, 64],
      "lstm_units": [25, 50],
      "fc_units": [15],
      "kernel_sizes": [3, 5],
      "dropout": 0.351789,
      "optimizer": ...
    }

#### TCN

    "model": {
      "type": "tcn",
      "list_n_filters": [32],
      "kernel_size": 4,
      "dilations": [1, 2, 4, 8],
      "nb_stacks": 3,
      "n_layers": 1,
      "activation": "norm_relu",
      "dropout_rate": 0.05,
      "use_skip_connections": true,
      "bidirectional": true,
      "optimizer": ...
    }

### Optimizers

New optimizers can be added in the file `smd/models/model_loader.py`.

Here are the already implemented optimizers with their configuration.

#### SGD + momentum

    "optimizer": {
      "name": "SGD",
      "lr": 0.001,
      "momentum": 0.9,
      "decay": 1e-6
    }

#### Adam

    "optimizer": {
      "name": "adam",
      "lr": 0.001,
      "beta_1": 0.9,
      "beta_2": 0.999,
      "epsilon": null,
      "decay": 0.0
    }

## Scripts

- `train.py` to start the training of an experiment, different things can be decided in this file like the data pre-processing and augmentation, the learning rate scheduler and the early stopping.
- `evaluate.py` to start the evaluation on the test set of a previously trained model. The pre-processing on the test set is decided in this file.
- `predict.py` to pass one file or folder through the network and save the output.
- `hyper_params_opt.py` to do the hyper-parameters optimization of a configuration with `hyperas`. Almost all the configuration for the hyper-parameters optimization has to be manually set in this file.
- `vizualize_data.py` to listen to the audio while visualizing the prediction and/or ground-truth with `sed_vis`.

## Real-time analysis of the audio

To analyze in real-time the audio taken from the microphone of the computer, one can try this [Github Project](https://github.com/qlemaire22/real-time-audio-analysis) that has been made to work with this framework.

You only need to put your trained model and the mean and std matrices of the dataset in the `model` folder of that project.

## References

[1] "Exploring Data Augmentation for Improved Singing Voice Detection with Neural Networks" by Jan Schlüter and Thomas Grill at the 16th International Society for Music Information Retrieval Conference (ISMIR 2015).
