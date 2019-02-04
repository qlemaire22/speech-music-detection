import matplotlib
matplotlib.use('TkAgg')

import smd.data.display.events as display

AUDIO_PATH = "./good_m.wav"
ANNOTATION_PATH = "./good.txt"

display.audio_with_events(AUDIO_PATH, ANNOTATION_PATH)
