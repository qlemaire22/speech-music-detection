import matplotlib
matplotlib.use('TkAgg')
import smd.data.display as display

AUDIO_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/ofai/audio_22050_600_100/332-RSR3-2012_02_01_10_45_01001.wav"
ANNOTATION_PATH = "/Users/quentin/Computer/DataSet/Music/speech_music_detection/ofai/audio_22050_600_100/332-RSR3-2012_02_01_10_45_01001.txt"

display.audio_with_events(AUDIO_PATH, ANNOTATION_PATH)
