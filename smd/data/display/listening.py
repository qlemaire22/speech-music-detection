from IPython.display import display, clear_output
import time
import numpy as np


def ipython_listen_with_labels(label, duration):
    # Not yet working
    n_frame = label.shape[1]
    duration_frame = duration / n_frame
    label = label.T

    for i in range(n_frame):
        clear_output(wait=True)
        if np.array_equal(label[i], [0, 0]):
            message = "time: " + str(int(i * duration_frame)) + " label: nothing"
        elif np.array_equal(label[i], [1, 0]):
            message = "time: " + str(int(i * duration_frame)) + " label: speech"
        elif np.array_equal(label[i], [0, 1]):
            message = "time: " + str(int(i * duration_frame)) + " label: music"
        else:
            message = "time: " + str(int(i * duration_frame)) + " label: speech and music"

        display(message)
        time.sleep(duration_frame)
