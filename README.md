# Speech & Music Detection

# !!! Still in development !!!

## Task

TODO: Task description

## Installation

The SoX command line utility is required ([HomePage](http://sox.sourceforge.net)).

Installation with HomeBrew:

    brew install lame
    brew reinstall sox --with-lame  # for mp3 compatibility

Installation of the Python packages with PyPi:

    pip install -r requirements.txt

TODO: Packages used

## Data

The different parameters of the data pre-processing or of the data augmentation can be changed in `smd/config.py`.

### Pre-processing

### Data Augmentation

Different transformations are applied to each training sample to do some data augmentation. The implemented transformations are:

- Time stretching
- Pitch shifting
- Dynamic range compression
- Gain changement
- Block mixing
- Background noise


## Experiment configuration

You can configure the model and the parameters of the training in the file `experiments.json`. Different models and optimizers can be choosen for this task.

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
