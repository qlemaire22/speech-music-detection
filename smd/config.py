# Preprocessing config

SAMPLING_RATE = 22050
FFT_WINDOW_SIZE = 1024
HOP_LENGTH = 512
N_MELS = 100
F_MIN = 27.5
F_MAX = 8000
AUDIO_MAX_LENGTH = 100  # in seconds

# Data augmentation config

STRETCHING_MIN = 0.75
STRETCHING_MAX = 1.25

SHIFTING_MIN = -3.5
SHIFTING_MAX = 3.5

GAIN_MIN = 0.5
GAIN_MAX = 1.5

BLOCK_MIXING_MIN = 0.1
BLOCK_MIXING_MAX = 0.9

# Model config

LOSS = "binary_crossentropy"
METRICS = ["categorical_accuracy"]
CLASSES = 2
