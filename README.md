# Speech & Music Detection

## Still in development

    sox test.wav 44100_sox.wav rate -v 44100
    pipreqs ./ --force

## Task

## Installation

The SoX command line utility is required ([HomePage](http://sox.sourceforge.net)).

Installation with HomeBrew:

    brew install sox

Installation of the Python packages with PyPi:

    pip install -r requirements.txt

TODO: Packages used

## Data

## Configuration

### Models

#### B-LTSM

    "model": {
      "type": "blstm",
      "hidden_units": [100, 100, 100],
      "dropout": 0.2,
      "optimizer": ...
    }

#### B-ConvLSTM

    "model": {
      "type": "bconvlstm",
      "filters_list": [50, 50],
      "kernel_size_list": [3, 3],
      "stride_list": [(1, 1), (1, 1)],
      "dilation_rate_list": [(1, 1), (1, 1)],
      "dropout": 0.2,
      "optimizer": ...
    }

#### TCN

    "model": {
      "type": "tcn",
      "nb_filters": 32,
      "kernel_size": 4,
      "dilations": [1, 2, 4, 8],
      "nb_stacks": 3,
      "activation"= "norm_relu",
      "use_skip_connections": true,
      "dropout_rate": 0.05,
      "optimizer": ...
    }

### Optimizers

#### SGD + momentum

    "optimizer": {
      "name": "SGD",
      "lr": 0.001,
      "momentum": 0.9,
      "decay": 1e-6
    }
