# Preprocessing config

SAMPLING_RATE = 22050
FFT_WINDOW_SIZE = 1024
HOP_LENGTH = 512
N_MELS = 80
F_MIN = 27.5
F_MAX = 8000
AUDIO_MAX_LENGTH = 90  # in seconds

# Data augmentation config

MAX_SHIFTING_PITCH = 0.3
MAX_STRETCHING = 0.3
MAX_LOUDNESS_DB = 10

BLOCK_MIXING_MIN = 0.2
BLOCK_MIXING_MAX = 0.5

# Model config

LOSS = "binary_crossentropy"
METRICS = ["binary_accuracy", "categorical_accuracy"]
CLASSES = 2

# For audio augmentation - deprecated

STRETCHING_MIN = 0.75
STRETCHING_MAX = 1.25

SHIFTING_MIN = -3.5
SHIFTING_MAX = 3.5
