import matplotlib
matplotlib.use('TkAgg')

import smd.data.display.events as display

AUDIO_PATH = "./total4_m.wav"
ANNOTATION_PATH = "./total4.txt"

display.audio_with_events(AUDIO_PATH, ANNOTATION_PATH)
